#!/usr/bin/env python
# coding: utf-8

# In[73]:


import json
import logging
import numpy as np
import pickle
import re
import os
import cem_classifier

from bs4 import BeautifulSoup
from chemdataextractor import Document
from chemdataextractor.doc import Paragraph
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

cwt = ChemWordTokenizer()

with open('/home/nlp/NN_word_classification/cem_classcification_model', 'rb') as f:
    mlp = pickle.load(f)
    
wv = Word2Vec.load('/home/nlp/word2vec_training/materials-word-embeddings/bin/word2vec_embeddings-SNAPSHOT.model').wv

# log setting
logging.basicConfig(format='%(levelname)s: %(funcName)s, %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# In[74]:


def name_patterns():
    pattern1 = re.compile(r"([\w\(\)\-\,\'\[\]\·]*)(?: or ([\w\(\)\-\,\'\[\]\·]*))? (?:was|were) (?:synthesi|obtain)")
    pattern2 = re.compile(r"(?:[sS]ynthes(?:is|es|ise)|[Pp]reparation)(?: and (?:C|c)haract[a-z]*)? (?:of|for) ([\w\(\)\-\,\'\[\]\·]*)(?: (\([\w\(\)\-\,\'\[\]\·]*\)))?")
    pattern3 = re.compile(r"(?:[sS]ynthes(?:is|es|ise)|[Pp]reparation)(?: and (?:C|c)haract[a-z]*)? ([\w\(\)\-\,\'\[\]\·]*)(?: (\([\w\(\)\-\,\'\[\]\·]*\)))?")
    MOF_pattern = re.compile(r"([\w\-\(\)]*(?:MOF|UiO|ZIF|HKUST|PCN|IRMOF|MIL|CPM)[\w\-\(\)]*)")

    pattern_groups = ((pattern1, 90, (1, 2)), (pattern2, 80, (1, 2)), (pattern3, 10, (1, 2)), (MOF_pattern, 20, (1,)))
    return pattern_groups

def extract_name(para, pattern_groups=name_patterns()):
    """
    extract product coumpound name from synthesis paragraph
    
    $para:
        para - str, paragraph
        patterns - triple, ((compiled_pattern1, confidence, indexes), (compiled_pattern2, confidence, indexes)), 
                   confidence for afterward filter, indexes for regex matched object index
    $return:
        result - str, name
    """
    name = {}
    
    # only extract from first two sentences
    doc = Paragraph(para)
    para = ''.join([str(sentence) for sentence in doc.sentences[:2]])
    
    for pattern_group in pattern_groups:
        pattern, confidence, indexes = pattern_group
        result = pattern.search(para)
        if result:
            for i in indexes:
                try:
                    # each match result saved with confidence for its pattern
                    if result.group(i):
                        if result.group(i) not in name:
                            name[result.group(i)] = confidence
                        else:
                            name[result.group(i)] += confidence
                except BaseException:
                    pass
                
    logging.info("name: {}".format(name))
    
    # only keep the result with highest confidence
    if name:
        name = nms_filter(name)
    else:
        name = ''
    
    name = re.sub(r"(\,|\:|\'|\")$",'',name)
    return name


# In[75]:


def temp_time_patterns():
    # The mixture was then heated in a 120 °C oven for 24 h.
    # The vial was heated in oil bath at 120 °C under stirring with a stir bar inside for 24 h. 
    # the mixture stirred for few minutes at room temperature, the autoclave sealed, placed in an oven at 220°C and kept at this temperature for 16 hours.
    # The autoclave was placed in an oven at 220°C for both syntheses and kept at this temperature for 6 hours (MIL-140B) and 12h (MIL-140C).</p>
    # The autoclave was placed in an oven at 180°C for both syntheses and kept at this temperature for 16 hours.</p>
    # and heated to 150 °C at a rate of 10 °C/h and then kept at 150 °C for 48 h.
    # The resulting solution was distributed among 6 Pyrex tubes (3 ml in each one) and placed into the oven (120 °C) for 24 hours.
    pattern1 = re.compile(r"(?:place|heat|stir|maintain|kept|held)[\w ,]*?\(?(?P<temp>(?:-?\d{1,3} ?(?:°C|K)|reflux|room temperature|refulxed))\)?[\w ,]*?for (?P<time>\d{1,3} ?(?:h|day|min))")
    
    # The glass reactor was sealed and heated under stirring for 15 min at 100 °C.
    # Water (3 mL) was then added under stirring, autoclave sealed and placed in the oven for 24 h at 130 °C.
    pattern2 = re.compile(r"(?:place|heat|stir|maintain|kept|held)[\w ,]*?(?P<time>\d{1,3} ?(?:h|day|min))[\w ,]*?at (?P<temp>(?:-?\d{1,3} ?(?:°C|K)|reflux|room temperature|refluxed))")
    
    #TODO
    #A solution of TPP-COOMe 0.854 g (1.0 mmol) and FeCl2·4H2O (2.5 g, 12.8 mmol) in 100 mL of DMF was refluxed for 6 h.
    #The two vials prepared were heated in a heating block on a hot plate, and the temperature of the heating block is set at 120 °C by using thermocouple. After 24 h, vials were taken out from the heating block, and the mother solution was removed by centrifuging (10 min, 7000 rpm).
    #The mixture was sonicated for 10 minutes. The solution was transferred to a Parr 45 mL acid digestion vessel, sealed and heated to 120 °C. After 2 days, white microcrystalline powder was obtained. The powder was collected by centrifugation and washed with clean DMF (3 × 20 mL).
    #Finally, the vial was placed into an oven and heated with a ramp rate of 0.2 °C/minute to 120 °C for 96 hours followed by cooling to 25 °C at a rate of 0.5 °C / minute.

    return {pattern1: 90, pattern2: 90}

def nms_filter(l):
    """
    Non maximum surpression filter
    
    $para:
        l - dict, {item1: confidence1, item2:confidence2}
    $return:
        result - item with highest confidence
    """
    l = sorted(list(l.items()), key=lambda x:x[1], reverse=True)
    return l[0][0]

def temp_normalize(temp):
    result = re.search(r"(\d{1,3}) ?°C", temp)
    if result:
        t = result.group(1)
        k = int(t) + 273
        temp = "{}K".format(k)
    return temp

def time_normalize(time):
    result = re.search(r"(\d{1,3}) ?(h|H|min|Min|d|D)", time)
    if result:
        t = result.group(1)
        unit = result.group(2)
        d = {'h': 1, 'H':1, 'min': 0.01666666666666666666666666666667, 'Min': 0.01666666666666666666666666666667, 'd': 24, 'D': 24}
        time = "{}h".format(str(int(t)*d[unit]))
    return time

def number_sub(para):
    number_sub = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
    for label in number_sub.keys():
        para = re.sub(r'{} ?(h|day|min)'.format(label),'{} \g<1>'.format(str(number_sub[label])),para)
    return para

def extract_temp_time(para, patterns=temp_time_patterns()):
    """
    extract temperature and time info from synthesis paragraph
    
    $para:
        para - str, paragraph
        patterns - dict, {compiled_pattern1: confidence, compiled_pattern2: confidence}, confidence for afterward filter
    $return:
        result - dict, {"temp": temp, "time":time}
    """
    temp = {}
    time = {}
    para = number_sub(para)
    for pattern, confidence in patterns.items():
        result = pattern.search(para)
        if result:
            try:
                # each match result saved with confidence for its pattern
                if result.group("temp") not in temp or temp[result.group("temp")] < confidence:
                    temp[result.group("temp")] = confidence
                if result.group("time") not in temp or time[result.group("time")] < confidence:
                    time[result.group("time")] = confidence
            except BaseException:
                pass
    # only keep the result with highest confidence
    # convert temperature into Kelvin (K), time into hour(h)
    logging.info("temperature: {}\ttime: {}".format(temp, time))
    if temp:
        temp = nms_filter(temp)
        temp = temp_normalize(temp)
    else:
        temp = ''
    if time:
        time = nms_filter(time)
        time = time_normalize(time)
    else:
        time = ''
    #logging.info("result temperature: {} time: {}".format(temp, time))
    
    return {"temp": temp, "time": time}


# In[76]:


def compound_patterns(compound):
    compound = compound.replace('(', '\(')
    compound = compound.replace(')', '\)')
    compound = compound.replace('[', '\[')
    compound = compound.replace(']', '\]')
    
    #74 mmol (15 g) of compound
    #125mL (118.5g, 1621mmol) of compound
    pattern1 = re.compile(r'(\d+(\.\d+)? ?m?u?(g|mol|L|l|M) \((\d+(\.\d+)? ?m?u?(g|mol|L|l|M), )?\d+(\.\d+)? ?m?u?(g|mol|L|l|M)\)) of {}'.format(compound))
    
    # 1.5 mL of 8 M compound
    # 1.5 mL 8 M compound
    pattern2 = re.compile(r'(\d+(\.\d+)? ?m?u?(g|mol|L|l|M)( of)? \d+(\.\d+)? ?m?u?(g|mol|L|l|M)) {}'.format(compound))
    
    # 60mg of compound (0.30mmol)
    pattern3 = re.compile(r'(\d+(\.\d+)? ?m?u?(g|mol|L|l|M) of {} \(\d+(\.\d+)? ?m?u?(g|mol|L|l|M)\))'.format(compound))
    
    # 35.0 mg compound
    # 35 mL compound
    # 35.0 mL compound
    # 0.35 g compound
    # 10.0 mL compound
    # 1.7 mL of compound
    pattern4 = re.compile(r'(\d+(\.\d+)? ?m?u?(g|mol|L|l|M))( of)? {}'.format(compound))
    
    # compound (no quantity) (35 mg, 35 mmol)
    # compound (no quantity) (35 mg)
    pattern5 = re.compile(r'{} \(([^ ]*?)\) \((\d+(\.\d+)? ?m?u?(g|mol|L|l|M)(,? ?\d+(\.\d+)? ?m?u?(g|mol|L|l|M))?)\)?'.format(compound))
    
    # compound (35 mg, 35 mmol)
    # compound (0.753 mg, 4 mmol; Sigma Aldrich, ≥98.0 %)
    pattern6 = re.compile(r'{} \((\d+(\.\d+)? ?m?u?(g|mol|L|l|M)(?:,|;)? ?\d+(\.\d+)? ?m?u?(g|mol|L|l|M))\)?'.format(compound))
    
    # compound (35 mg)
    # compound (35.25 mg)
    # compound (35 mM)
    # compound (35 mmol)
    # compound (35mg)
    # compound (35.25mg)
    # compound (35mM)
    # compound (35mmol)
    # compound (18 uL, 5 equiv.)
    pattern7 = re.compile(r'{} \((\d+(\.\d+)? ?m?u?(g|mol|L|l|M))\)?'.format(compound))
    
    #compound (xxx, 300mg)
    #compound (xxx, 300 mg)
    pattern8 = re.compile(r'{} \([^,]*, (\d+(\.\d+)? ?m?u?(g|mol|L|l|M))\)'.format(compound))
    
    #15.0g (64.4 mmol) compound
    pattern9 = re.compile(r'((\d+(\.\d+)? ?m?u?(g|mol|L|l|M)) \((\d+(\.\d+)? ?m?u?(g|mol|L|l|M))\)) {}'.format(compound))
    
    # (1.600 g, 13.11 mmol) compound
    pattern10 = re.compile(r'\((\d+(\.\d+)? ?m?u?(g|mol|L|l|M)(?:,|;)? ?\d+(\.\d+)? ?m?u?(g|mol|L|l|M))\) {}'.format(compound))
    
    
    
    return [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10]


def quantity_classifier(quantity):
    
    quantity_info = {'mass':'','volume':'','concentration':'','mol':''}
    results = re.findall(r'(\d+(\.\d+)?) ?(m?u?)(g|mol|L|l|M)',quantity)
    dic = {'':1000,'m':1,'u':0.001}
    if results:
        for result in results:
            q = result[0]
            mu = result[2]
            u = result[3]
            normalized_quantity = '{}m'.format(str(round(float(q)*dic[mu],4))) + u
            if u == 'g':
                quantity_info['mass'] = normalized_quantity
            elif u == 'L' or u == 'l':
                quantity_info['volume'] = normalized_quantity.replace('ml','mL')
            elif u == 'mol':
                quantity_info['mol'] = normalized_quantity
            elif u == 'M':
                quantity_info['concentration'] = normalized_quantity
    
    return [i[1] for i in list(quantity_info.items())]


def extract_quantity(cem, para):
    """
    extract quantity info by cem name from synthesis paragraph
    
    $para:
        cem - str, name of chemical entity 
        para - str, paragraph with label on cem categories, eg. <element category="linker">H3BPY</element>
    $return:
        result - dict, [cem, '10mg, '10mL', '', '']
    """
    cem_list = [cem]
    patterns=compound_patterns(cem)
    for i, pattern in enumerate(patterns):
        if i == 4:
            results = pattern.findall(para)
            if results:
                logging.info("pattern: {}".format(i+1))
                for result in results:
                    quantity_pattern = re.compile(r"\d+(\.\d+)? ?m?u?(g|mol|L|l|M)")
                    quantity_result = quantity_pattern.search(result[0])
                    if not quantity_result or result[0] == 'DEF':
                        cem_list.extend(quantity_classifier(result[1]))
                        logging.info('{}'.format(cem_list))
                        return cem_list
            
        else:
            results = pattern.findall(para)
            if results:
                logging.info("pattern: {}".format(i+1))
                for result in results:
                    cem_list.extend(quantity_classifier(result[0]))
                    logging.info('{}'.format(cem_list))
                    return cem_list
    return [cem,'','','','']


# In[77]:


# para = "In a typical procedure of UiO-66-NH2-Fast, 15.0 g (64.4 mmol) ZrCl4, 11.7 g (64.4 mmol) 2-aminoterephthalic acid (BDC-NH2) and 440 mL (7.73 mol) acetic acid were dissolved in 1 L DMF in a 2 L three-necked flask, and then 75 mL H2O was added. The resulting homogeneous solution was heated in an oil bath under stirring at 120 °C for 15 min before it was cooled to room temperature. The product was separated via centrifugation at 10000 rpm for 3 minutes and further purified with ethanol several times."
# para2 = "4,4',4'',4'''-(9,9'-Spirobi[fluorene]-2,2',7,7'-tetrayl)tetrabenzoic acid (3) (24 mg, 0.03 mmol) and ZrOCl2·8H2O (40 mg, 0.12 mmol) was dissolved in DMF (8 mL) following addition of formic acid (1 mL) and the solution was sonicated for 10 min."
# para3 = "70 mg of ZrCl4 (0.30 mmol) and 2700 mg (22 mmol) of benzoic acid were mixed in 8 mL of DMF (in a 6-dram vial) and ultrasonically dissolved."
# para4 = "In a typical procedure of Pd@UiO-66-NH2-Fast, 1.0 g (4.3 mmol) ZrCl4, 0.78 g (4.3 mmol) BDC-NH2 and 30 mL (0.52 mol) acetic acid were dissolved in 160 mL DMF in a 500 mL three-necked flask,"
# extract_quantity("acetic acid", para4)


# In[78]:


def get_vector(token, wv=wv):
    vector = np.zeros((100,))
    
    if token in wv:
        vector = wv[token]
    else:
        for word in cwt.tokenize(token):
            try:
                vector += wv[word]
            except:
                logging.info('{} from {} can\'t be found'.format(word, token))
    return vector

def label_cems(para, cems_and_labels):
    labeled_para = ''
    for i in range(len(cems_and_labels)):
        cem = cems_and_labels[i][0]
        label = cems_and_labels[i][1]
        # first segment
        if i == 0:
            if cem.start != 0:
                labeled_para += para[:cem.start]

        labeled_para += '<element category=\"{}\">'.format(label) + para[cem.start: cem.end] + '</element>'
        
        if i+1 < len(cems_and_labels):
            next_cem = cems_and_labels[i+1][0]
            labeled_para += para[cem.end: next_cem.start]
        else:
            # last segment
            labeled_para += para[cem.end: ]

    return labeled_para
        
def sentence_patterns():
    pattern1 = re.compile(r"wash|purified|clean|remove|replace|digest|dried|drying|centrifuged|centrifuging|exchange|metalate|suspend")
    pattern2 = re.compile(r"[Cc]alculated|Calcd|cal(cd)?\(%\)|(FT)?IR|[Ff]ound|EA data|[Ee]lemental analysis|NMR|ESI-MS|space group")
    return [pattern1,pattern2]

def find_sentence(cem,para):
    para = Paragraph(para)
    for sentence in para.sentences:
        if sentence.start <= cem.start and sentence.end >= cem.end:
            return sentence.text
    

def cem_filter(cem,para,patterns = sentence_patterns()):
    # remove Teflon
    teflon_pattern = re.compile(r"[Tt]eflon")
    if teflon_pattern.search(cem.text):
        return False
    
    # remove PTFE
    PTFE_pattern = re.compile(r"PTFE")
    if PTFE_pattern.search(cem.text):
        return False
    
    #remove MOF names
    MOF_pattern = re.compile(r"([\w\-\(\)]*(?:MOF|UiO|ZIF|HKUST|PCN|IRMOF|MIL|CPM)[\w\-\(\)]*)")
    if MOF_pattern.search(cem.text):
        return False
    
    #remove cems in sentences other than synthesis
    sentence = find_sentence(cem, para)
    for pattern in patterns:
        if pattern.search(sentence):
            logging.info('cem: {}\nsentence: {}\npattern: {}'.format(cem,sentence,pattern))
            return False
    
    return True

def extract_cems_and_label_para(para, name=None, mlp=mlp):
    labels = {0: 'metal-source',1: 'linker',2: 'solvent',3: 'modulator',4: 'temp',5: 'time',6: 'name',7: 'others'}
    
    doc = Document(para)
    cems_and_labels = []
    cems = sorted(list(doc.cems), key=lambda x:x.start)
    for cem in cems:
        if cem_filter(cem,para):
            if cem.text != name:
                label = cem_classifier.cem_classifier(cem.text)
                cems_and_labels.append([cem, label])
    
    logging.info('cems: {}'.format([[i[0].text, i[1]] for i in cems_and_labels]))
    
    labeled_para = label_cems(para, cems_and_labels)
    return labeled_para


# In[79]:


# para = "In a typical procedure, 1.5 g (4.7 mmol) HfCl4, 0.78 g (4.7 mmol) BDC and 12 mL acetic acid were dissolved in 200 mL DMF in a 500 mL three-necked flask, and then 5 mL H2O was added."
# para1 = "In a typical procedure of Pd@UiO-66-NH2-Fast, 1.0 g (4.3 mmol) ZrCl4, 0.78 g (4.3 mmol) BDC-NH2 and 30 mL (0.52 mol) acetic acid were dissolved in 160 mL DMF in a 500 mL three-necked flask,"
# extract_cems_and_label_para(para1)


# In[80]:


def find_labeled_sentence(labeled_cem,labeled_para):
    start = labeled_para.find(labeled_cem)
    end = start + len(labeled_cem)
    labeled_para = Paragraph(labeled_para)
    for sentence in labeled_para.sentences:
        if start >= sentence.start and end <= sentence.end:
            return sentence.text


# In[81]:


def extract_labeled_para(labeled_para, doi):
    
    extract_result = {}

    para = re.sub(r'<\/?element.*?>', '', str(labeled_para))
    logging.info('\n{}'.format(para))
    extract_result['text'] = para
    extract_result['doi'] = doi
    
    # extract name
    extract_result['name'] = extract_name(para)
    
    # extract time and temperature
    temp_time_result = extract_temp_time(para)
    extract_result['temperature'] = temp_time_result['temp']
    extract_result['time'] = temp_time_result['time']

    # extract cems and quantity
    extract_result['metal-source'] = []
    extract_result['linker'] = []
    extract_result['modulator'] = []
    extract_result['solvent'] = []
    extract_result['others'] = []
    
    extract_result['labeled_text'] = labeled_para

    soup = BeautifulSoup(labeled_para, "lxml")
    elements = []
    for element in soup.find_all('element'):
        if element not in elements:
            elements.append(element)
        else:
            continue
        #logging.info('to extract_quantity, {}, {}'.format(element.string, element))
        labeled_sentence = find_labeled_sentence(str(element),labeled_para)
        sentence = re.sub(r"<\/?element.*?>", '', labeled_sentence)
        quantity = extract_quantity(element.string, sentence)
        extract_result[element['category']].append(quantity)

    return extract_result


# In[82]:


def abbr_filter(abbreviations):
    """
    check if abbreviations is valid by send query to Pubchem
    
    $param:
        abbreviations - list of words, ([abbreviation], [original text], POS)
    
    $return:
        abbreviations - list of words, ([abbreviation], [original text], POS)
    """
    abbrs_filtered = []

    for abbr in abbreviations:
        # if POS of abbr is 'CM', then no need to check using pubchem
        if abbr[2] == 'CM':
            abbrs_filtered.append(abbr)
        else:
            pass
        
    return abbrs_filtered

def get_abbr(fname):
    with open(fname, 'rb') as f:
        doc = Document.from_file(f)
        abbrs = doc.abbreviation_definitions
        abbrs_filtered = abbr_filter(abbrs)
        
#     logging.info('fname: {}'.format(fname))
#     logging.info('abbrs_filterd:\n{}'.format('\n'.join([str(i) for i in abbrs_filtered])))
    
    return abbrs_filtered


# In[83]:


def get_abbr_info(para,abbr,start):
    full_name = " ".join(abbr[1])
    new_abbr = {"abbreviation":abbr[0][0],"full_name":full_name,"start":0,"end":0}
    para = Paragraph(para)
    for sentence_token in para.tokens:
        for token in sentence_token:
            if token.text == new_abbr["abbreviation"] and token.start >= start:
                #logging.info("token.start: {} token.end: {}".format(token.start,token.end))
                new_abbr["start"] = token.start
                new_abbr["end"] = token.end
                break
    #logging.info("abbr_info: {}".format(new_abbr))
    if new_abbr["end"] == 0:
        return False
    else:
        return new_abbr

def clean_repetition(abbrs):
    cleaned_abbrs = []
    for abbr in abbrs:
        if not abbr in cleaned_abbrs:
            cleaned_abbrs.append(abbr)
    
    return cleaned_abbrs
    
def find_abbrs(para,doi):
    para_abbrs = []
    doi = doi.replace("/","_")
    if doi.startswith("SI_"):
        txt_doi = doi[3:]
        si_doi = doi
    else:
        txt_doi = doi
        si_doi = "SI_" + doi
    
    txt_fname = "/home/nlp/Pipeline/Zr_MOF/txt/{}.txt".format(txt_doi)
    si_fname = "/home/nlp/Pipeline/Zr_MOF/si_txt/{}.txt".format(si_doi)
    abbrs = get_abbr(txt_fname)
    try:
        si_abbrs = get_abbr(si_fname)
        abbrs.extend(si_abbrs)
    except:
        pass
    abbrs = clean_repetition(abbrs)
    #logging.info("abbrs in text: {}\n".format(abbrs))
    
    for abbr in abbrs:
        pattern = re.compile(r"{}".format(abbr[0][0]))
        results = pattern.findall(para)
        #logging.info("find result: {}\n".format(results))
        if results:
            start = 0
            for result in results:
                abbr_info = get_abbr_info(para,abbr,start)
                if abbr_info:
                    para_abbrs.append(abbr_info)
                    start = abbr_info["end"]
            
    sorted_para_abbrs = sorted(para_abbrs, key=lambda x:x['end'])
    
    return sorted_para_abbrs 



def sub_abbrs(para,doi):
    abbrs = find_abbrs(para,doi)
    if not abbrs:
        return para
    #logging.info("abbrs in para: {}\n".format(abbrs))
    subed_para = ""
    for i in range(len(abbrs)):
        if abbrs[i]["end"] == 0:
            continue            
        # first segment
        if i == 0:
            if abbrs[i]["start"] != 0:
                subed_para += para[:abbrs[i]["start"]]

        subed_para += abbrs[i]["full_name"]
        
        # not last segment?
        if i+1 < len(abbrs):
            subed_para += para[abbrs[i]["end"]: abbrs[i+1]["start"]]
        # last segment
        else:
            subed_para += para[abbrs[i]["end"]: ]

    return subed_para


# In[84]:


# para1 = "<p category='Non'>The Tetrakis(triphenylphosphine)palladium(0) was purchased from Pressure Chemical Co. The Cesium fluoride (CsF), dimethoxyethane (DME), N,N'-dimethylformamide (DMF), N,N'-dimethylacetamide (DMA), N,N'-diethylformamide (DEF), chloroform (CHCl3), trifluoroacetic acid (CF3COOH), benzoic acid, acetone, zirconium(IV) chloride (ZrCl4), hafnium chloride (HfCl4) tetrahydrofuran (THF) and methanol were purchased from VWR. All commercial chemicals were used without further purification unless otherwise mentioned. 1H nuclear magnetic resonance (NMR) data were recorded on a Mercury 300 MHz NMR spectrometer at the Center for Chemical Characterization and Analysis (CCCA), Department of Chemistry, Texas A&M University. Fourier transform infrared spectroscopy (FTIR) data were collected using a SHIMADZU IRAffinity-1 FTIR Spectrophotometer.</p>"

# para2 = "wash with DME, DME and DME"

# para3 = "wash with DME"



# para = Paragraph(para2)
# for sentence in para.tokens:
#     for token in sentence:
#         if token.text == "DME":
#             print(token.start)

# pattern = re.compile(r"DMF")
# result = pattern.findall(para2)
# print(result)
# sub_abbrs(para2,"10.1002_ange.201307340")
# find_abbrs(para2,"10.1002_ange.201307340")


# In[85]:


def get_abbr_in_para(para):
    para = Paragraph(para)
    abbrs = para.abbreviation_definitions
    abbrs_filtered = abbr_filter(abbrs)
        
#     logging.info('fname: {}'.format(fname))
#     logging.info('abbrs_filterd:\n{}'.format('\n'.join([str(i) for i in abbrs_filtered])))
    
    return abbrs_filtered

def find_abbrs_in_para(para):
    para_abbrs = []
    abbrs = get_abbr_in_para(para)
    abbrs = clean_repetition(abbrs)
    #logging.info("abbrs in text: {}\n".format(abbrs))
    
    for abbr in abbrs:
        pattern = re.compile(r"{}".format(abbr[0][0]))
        results = pattern.findall(para)
        #logging.info("find result: {}\n".format(results))
        if results:
            start = 0
            for result in results:
                abbr_info = get_abbr_info(para,abbr,start)
                if abbr_info:
                    para_abbrs.append(abbr_info)
                    start = abbr_info["end"]
            
    sorted_para_abbrs = sorted(para_abbrs, key=lambda x:x['end'])
    
    return sorted_para_abbrs 

def sub_abbrs_in_para(para):
    abbrs = find_abbrs_in_para(para)
    logging.info("abbrs in para: {}\n".format(abbrs))
    if not abbrs:
        return para
    subed_para = ""
    for i in range(len(abbrs)):
        if abbrs[i]["end"] == 0:
            continue            
        # first segment
        if i == 0:
            if abbrs[i]["start"] != 0:
                subed_para += para[:abbrs[i]["start"]]

        subed_para += abbrs[i]["full_name"]
        
        # not last segment?
        if i+1 < len(abbrs):
            subed_para += para[abbrs[i]["end"]: abbrs[i+1]["start"]]
        # last segment
        else:
            subed_para += para[abbrs[i]["end"]: ]
            
    for abbr in abbrs:
        subed_para = re.sub(r"{} \({}\)".format(abbr["full_name"],abbr["full_name"]),"{}".format(abbr["full_name"]),subed_para)

    return subed_para


# In[86]:


# s = "MIL-140D or ZrO[O2C-C12N2H6Cl2-CO2] was synthesized in a similar manner as MIL-140A with a 23mL Teflon lined steel autoclave starting from 1mmol (340 mg) of 3,3'-dichloro-4,4'azobenzenedicarboxylic acid (Cl2AzoBDC), 0.5mmol (117mg) of ZrCl4 (Alfa Aesar, 99.5+%), 285uL (299mg, 5mmol) of acetic acid (CH3CO2H) and 5mL (4.75g, 65mmol) of DMF. The autoclave was placed in an oven at 180°C for both syntheses and kept at this temperature for 16 hours."
# s2 = "Gallic acid monohydrate (C7H6O5.H2O) (0.753 g, 4 mmol; Sigma Aldrich, ≥98.0 %) was dissolved in N,N-diethylformamide (DEF) (5 ml) at room temperature. ZrCl4 (0.233 g, 1 mmol; Alfa Aesar, 99.5+ %) was then added to the solution. The reaction mixture was then sealed and placed in an oven, and heated to 180 °C for 24 hours."
# print(sub_abbrs_in_para(s2))


# In[87]:


def extract_para(para, doi):
    
    
    extract_result = {}

    logging.info('\n{}'.format(para))
    extract_result['text'] = para
    
    #replace abbrs
    para = sub_abbrs_in_para(para)
    # replace water with H2O and DEF with Diethylacetamide
    para = re.sub(r" [Ww]ater([ ,.])"," H2O\g<1>",para)
    para = re.sub(r" DEF([ ,.])"," Diethylacetamide\g<1>",para)
    
    extract_result['doi'] = doi.replace("_","/")
    
    # extract name
    extract_result['name'] = extract_name(para)
    
    # extract time and temperature
    temp_time_result = extract_temp_time(para)
    extract_result['temperature'] = temp_time_result['temp']
    extract_result['time'] = temp_time_result['time']

    # extract cems and quantity
    extract_result['metal-source'] = []
    extract_result['linker'] = []
    extract_result['modulator'] = []
    extract_result['solvent'] = []
    extract_result['others'] = []
    
    labeled_para = extract_cems_and_label_para(para, name=extract_result['name'])
    extract_result['labeled_text'] = labeled_para
    
    soup = BeautifulSoup(labeled_para, "lxml")
    elements = []
    for element in soup.find_all('element'):
        if element not in elements:
            elements.append(element)
        else:
            continue
        
        labeled_sentence = find_labeled_sentence(str(element),labeled_para)
        sentence = re.sub(r"<\/?element.*?>", '', labeled_sentence)
        quantity = extract_quantity(element.string, sentence)
        if quantity == [element.string,'','','','']:
            logging.info("empty quantity: {}".format(element.string))
            continue
        extract_result[element['category']].append(quantity)

    return extract_result


# In[91]:


# para1 = "In a typical procedure of UiO-66-NH2-Fast, 15.0 g (64.4 mmol) ZrCl4, 11.7 g (64.4 mmol) 2-aminoterephthalic acid (BDC-NH2) and 440 mL (7.73 mol) acetic acid were dissolved in 1 L DMF in a 2 L three-necked flask, and then 75 mL H2O was added. The resulting homogeneous solution was heated in an oil bath under stirring at 120 °C for 15 min before it was cooled to room temperature. The product was separated via centrifugation at 10000 rpm for 3 minutes and further purified with ethanol several times."
# extract_para(para,"1")


# In[89]:


import unittest

class TestExtractor(unittest.TestCase):
    
    def test_time_temp_extractor(self):
        d = {"The mixture was then heated in a 120 °C oven for 24 h.": {'temp': '393K', 'time': '24h'},
             "The resulting solution was distributed among 6 Pyrex tubes (3 ml in each one) and placed into the oven (120 °C) for 24 hours.": {'temp': '393K', 'time': '24h'}, 
             "The autoclave was placed in an oven at 180°C for both syntheses and kept at this temperature for 16 hours.": {'temp': '453K', 'time': '16h'}, 
             "The glass reactor was sealed and heated under stirring for 15 min at 100 °C.": {'temp': '373K', 'time': '0.25h'}}
        for q, a in d.items():
            self.assertEqual(extract_temp_time(q), a)

    def test_cems_extractor(self):
        q = "Synthesis of TPHN-MOF. ZrCl4 (10 mg) and H2TPHN (20 mg) were dissolved in 10 mL of DMF in a 5 dram vial, and 0.1 mL of trifluoroacetic acid was added. The solution was then heated at 100 °C for 5 days to afford small bright yellow crystals as the MOF product (yield: 21 mg, 84%)."
        a = "Synthesis of TPHN-MOF. <element category='metal-source'>ZrCl4</element> (10 mg) and <element category='linker'>H2TPHN</element> (20 mg) were dissolved in 10 mL of <element category='solvent'>DMF</element> in a 5 dram vial, and 0.1 mL of <element category='modulator'>trifluoroacetic acid</element> was added. The solution was then heated at 100 °C for 5 days to afford small bright yellow crystals as the MOF product (yield: 21 mg, 84%)."
        self.assertEqual(extract_cems_and_label_para(q), a)
        
    def test_quantity_extractor(self):
        d = {"74 mmol (15 g) of compound": ['compound', '15000.0mg', '', '', '74.0mmol'], 
             "1.5 mL of 8 M compound": ['compound', '', '1.5mL', '8000.0mM', ''], 
             "0.35 g compound": ['compound', '350.0mg', '', '', ''], 
             "compound (35 mg, 35 mmol)": ['compound', '35.0mg', '', '', '35.0mmol'], 
             "compound (35 mmol)": ['compound', '', '', '', '35.0mmol'], 
             "compound (xxx, 300mg)": ['compound', '300.0mg', '', '', '']}
        for q, a in d.items():
            self.assertEqual(extract_quantity('compound', q), a)


# In[90]:


if __name__ == '__main__':
    unittest.main()


# In[ ]:




