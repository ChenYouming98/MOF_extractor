#!/usr/bin/env python
# coding: utf-8


import logging
import json
import numpy as np
import os
import pickle
import re
import requests
import traceback
import urllib.request

from bs4 import BeautifulSoup
from chemdataextractor import Document
from chemdataextractor.reader import acs, rsc
from wiley_reader import wiley_reader
from paragraph_classifier import load_classifier_model, predict_if_synthese
from articledownloader.articledownloader import ArticleDownloader
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.converter import PDFPageAggregator

import pattern_based_extractor


# log setting
logging.basicConfig(format='%(levelname)s: %(funcName)s, %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

downloader = ArticleDownloader(
    els_api_key='f1916da20a304f17ce6544b55b7c67bd', sleep_sec=30, timeout_sec=30)


def get_publisher(doi, doi_cache=None):
    """
    get publisher info of paper according to doi
    
    $Param:
        doi - str, doi of paper
    
    $return:
        publisher - str, name of publisher (['acs', 'rsc', 'wiley', 'springer', 'nature', elsevier', 'unkonw']).
    """
    logging.debug('doi: %s' % doi)
    publisher = None
    
    if not doi_cache:
        with open('doi_cache.json', 'r') as f:
            doi_cache = json.load(f)

    if doi[3:7] in doi_cache:
        publisher = doi_cache[doi[3:7]]
        return publisher
    else:
        try:
            url = 'http://dx.doi.org/' + doi
            headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, "html.parser")

            if soup.find(content=re.compile("pubs.acs.org")):
                publisher = 'acs'
            elif soup.find(content=re.compile("pubs.rsc.org")):
                publisher = 'rsc'
            elif soup.find(content=re.compile("onlinelibrary.wiley.com")):
                publisher = 'wiley'
            elif soup.find(content=re.compile("link.springer.com")):
                publisher = 'springer'
            elif soup.find(content=re.compile("www.nature.com")):
                publisher = 'nature'
            elif soup.find(content=re.compile("www.sciencedirect.com")):
                publisher = 'elsevier'
            elif soup.find(text="DOI Not Found"):
                publisher = 'error'
            else:
                publisher = 'unknow'
            logging.info('doi: %s, publisher: %s' % (doi, publisher))
        except BaseException as e:
            logging.error('doi: %s, publisher: %s, error: %s' % (doi, publisher, e))
            publisher = 'error'
        finally:
            if publisher not in ['error', 'unknow']:
                doi_cache[doi[3:7]] = publisher
                with open('doi_cache.json', 'w') as f:
                    json.dump(doi_cache, f)
            return publisher
        

def download_article(dir_save, doi, mode=None, verbose=0):
    """
    download html or xml format paper with name of {doi}.html or {doi}.xml
    
    dir_save - str, path to save html file
    doi - list, doi
    mode - str, pulisher of paper
    verbose - int, 0 for only print download failed info, 1 for print all download info
    
    return - 1 if success, 0 if failed
    """
    try:
        if mode == None:
            mode = get_publisher(doi)
        # 由于doi中含'/', 不适合作为文件名, 因此把'/'改为'-'作为文件名
        filename = '{}/{}'.format(dir_save, doi.replace('/','_'))
        #修改后的rsc文章下载方法
        if mode == 'rsc':
            url = 'http://dx.doi.org/' + doi
            headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, "html.parser")
            fulltext_html_url = soup.find(attrs={"name":"citation_fulltext_html_url"})['content']
            html = urllib.request.urlopen(fulltext_html_url).read()
            with open('{}.html'.format(filename), 'wb') as f:
                f.write(html)
        elif mode == 'elsevier':
            with open('{}.xml'.format(filename), 'wb') as f:
                downloader.get_xml_from_doi(doi, f, mode=mode)
        else:
            with open('{}.html'.format(filename), 'wb') as f:
                downloader.get_html_from_doi(doi, f, mode=mode)
        if verbose == '1':            
        	logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
        return 1
    except BaseException as e:
        logging.error('failed. doi: {}, publisher: {}, error: {}'.format(doi, mode, e))
        return 0
    

def download_si(dir_save, doi, mode=None, verbose=0):
    filename = '{}/SI_{}'.format(dir_save, doi.replace('/','_'))

    try:
        url = 'http://dx.doi.org/' + doi
        headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=(100))
        soup = BeautifulSoup(r.content, "html.parser")
        
        if mode == None:
            mode = get_publisher(doi)

        # ACS
        if mode == 'acs':
            r = requests.get(''.join(['https://pubs.acs.org/doi/suppl/', doi]), timeout=(60, 60))
            soup = BeautifulSoup(r.content, "html.parser")
            pdf = str(soup.find(text=re.compile("\.pdf")).find_parents("a"))
            pdf = ''.join(re.findall('href="(.*\.pdf)"', pdf))
            pdf = 'https://pubs.acs.org' + pdf
            r1 = requests.get(pdf, timeout=(100))

        # RSC
        elif mode == 'rsc':
            pdf = str(soup.find(href=re.compile(".pdf")))
            pdf = ''.join(re.findall('href="(.*\.pdf)"', pdf))
            r1 = requests.get(pdf, timeout=(100))

        # Wiley
        elif mode == 'wiley':
            pdf = str(soup.find(class_="support-info__table-wrapper article-table-content-wrapper"))
            pdf = ''.join(re.findall('href="(.*\.pdf)"', pdf))
            if pdf[0:4] == 'http':
                pass
            else:
                pdf = 'https://onlinelibrary.wiley.com' + pdf
            if pdf.find('&amp;') != -1:
                pdf = pdf[0:pdf.find('&amp;') + 1] + pdf[pdf.find('&amp;') + 5:]
            r1 = requests.get(pdf, timeout=(100))

        # Springer
        elif mode == 'springer' :
            pdf = str(soup.find(text=re.compile(".pdf")).find_parents("a"))
            pdf = ''.join(re.findall('href="(.*\.pdf)"', pdf))
            r1 = requests.get(pdf, timeout=(100))
            
        # Nature
        elif mode == 'nature':
            pdf = str(soup.find(href=re.compile(".pdf")))
            pdf = ''.join(re.findall('href="(.*\.pdf)"', pdf))
            r1 = requests.get(pdf, timeout=(100))
            doi = doi.replace('/', '_')

        # Elsevier
        elif mode == 'elsevier':
            xml_url='https://api.elsevier.com/content/article/doi/' + doi + '?view=FULL'
            headers = {
              'X-ELS-APIKEY': 'f1916da20a304f17ce6544b55b7c67bd',
              'Accept': 'text/xml'
            }
            r = requests.get(xml_url, stream=True, headers=headers, timeout=(100))
            soup = BeautifulSoup(r.content,'xml')
            attachments = soup.find_all('xocs:attachment-eid')
            findpdf = re.compile(r'(<xocs:attachment-eid>)([\S]+mmc\d\.pdf)(<\/xocs:attachment-eid>)')
            finddoc = re.compile(r'(<xocs:attachment-eid>)([\S]+mmc\d\.doc)(<\/xocs:attachment-eid>)')
            finddocx = re.compile(r'(<xocs:attachment-eid>)([\S]+mmc\d\.docx)(<\/xocs:attachment-eid>)')
            for attachment in attachments:
                pdf = findpdf.search(repr(attachment))
                docx = finddocx.search(repr(attachment))
                doc = finddoc.search(repr(attachment))
                if pdf:
                    r1 = requests.get('https://ars.els-cdn.com/content/image/' + pdf.group(2), timeout=(100))
                elif docx:
                    r2 = requests.get('https://ars.els-cdn.com/content/image/' + docx.group(2), timeout=(100))
                elif doc:
                    r3 = requests.get('https://ars.els-cdn.com/content/image/' + doc.group(2), timeout=(100))
    
        if 'r1' in locals():
            with open('{}.pdf'.format(filename), 'wb') as f:
                f.write(r1.content)
            if verbose == '1':            
        		logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
            return 1
        elif 'r2' in locals():
            with open('{}.docx'.format(filename), 'wb') as f:
                f.write(r2.content)
            if verbose == '1':            
        		logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
            return 1
        elif 'r3' in locals():
            with open('{}.doc'.format(filename), 'wb') as f:
                f.write(r3.content)
            if verbose == '1':            
        		logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
            return 1
        else:
            logging.info('failed. doi: {}, publisher: {}, error: {}'.format(doi, mode, 'no SI'))
            return 0
            
    except BaseException as e:
        logging.error('failed. doi: {}, publisher: {}, error: {}'.format(doi, mode, e))
        return 0


def xml_to_txt(fname_in, fname_out=None, doi=None, publisher='unknow', verbose=0):
    """
    convert xml format paper into txt format
    
    $Param:
        fname_in - str, path to html file
        fname_out - str, path to output txt file
        doi - str, doi of paper, if not present, then try extract from filename
        publisher - str, pulisher of paper, if not present, then try get from doi
        verbose - int, 0 for no output, 1 for print success info
    
    $Return:
        int, 1 if success, 0 if failed
    """
    if not fname_out:
        fname_out = '{}.txt'.format(fname_in[:-4])
        
    if not doi:
        doi = os.path.basename(fname_in[:-4]).replace('_', '/')
        
    try:
        with open(fname_in,'r', encoding='UTF-8') as f1, open(fname_out, 'w') as f2:
            soup = BeautifulSoup(f1,'xml')
            paras_with_tags = soup.find_all('ce:para')
            abstract_with_tags = soup.find_all('ce:abstract')
            findtags = re.compile("<.+?>")
            
            for p in abstract_with_tags:
                para = findtags.sub('', repr(p))
                para = re.sub(r'\n', '', para)
                para = re.sub('&lt;', '<', para)
                para = re.sub('&gt;', '>', para)
                para = re.sub('\[[\d–,]+\]', '', para)
                f2.write('<p>')
                f2.write(para.strip('\n'))
                f2.write('</p>')
                f2.write('\n')
            
            for p in paras_with_tags:
                para = findtags.sub('', repr(p))
                para = re.sub(r'\n', '', para)
                para = re.sub('\[[\d–,]+\]', '', para)
                para = re.sub('&lt;', '<', para)
                para = re.sub('&gt;', '>', para)
                f2.write('<p>')
                f2.write(para.strip('\n'))
                f2.write('</p>')
                f2.write('\n')

        if verbose == '1':            
    		logging.info('success. doi: {}, publisher: {}'.format(doi, mode))

        return 1
    
    except BaseException as e:
        logging.error('failed. filename: {}, doi: {}, error: {}'.format(fname_in, doi, e))
        return 0


def judge_article_content(tag):
    if(tag.parent.has_attr('class')):
        parent = tag.parent
        parent_class = str(parent['class'])
        return parent_class.startswith('[\'article') or parent_class.endswith('abstract\']')
    else:
        return False


def wiley_reader(fname_in):
    with open(fname_in,'r', encoding='UTF-8') as f:
        soup = BeautifulSoup(f,'lxml')

    p_tags = soup.find_all('p')

    p_article_tags=[]

    for p in p_tags:
        if(judge_article_content(p)):
            p_article_tags.append(p)
        else:
            continue

    for p in p_article_tags:
        num_span = len(p.find_all('span'))
        count = 0
        while(count<num_span):
            p.span.decompose()
            count = count+1

    for p in p_article_tags:
        num_a = len(p.find_all('a'))
        count = 0
        while(count<num_a):
            p.a.decompose()
            count = count+1

    str_p = []
    for i in p_article_tags:
        str_p.append(repr(i))

    count = 0
    for i in str_p:
        str_p[count] = str_p[count].replace('<i>','')
        str_p[count] = str_p[count].replace('</i>','')
        str_p[count] = str_p[count].replace('<sub>','')
        str_p[count] = str_p[count].replace('</sub>','')
        str_p[count] = str_p[count].replace('<sup>','')
        str_p[count] = str_p[count].replace('</sup>','')
        str_p[count] = str_p[count].replace('<b>','')
        str_p[count] = str_p[count].replace('</b>','')
        str_p[count] = str_p[count].replace('\n','')
        str_p[count] = str_p[count].replace('[]','')
        str_p[count] = str_p[count].replace('[, ]','')
        str_p[count] = str_p[count].replace('<p>','')
        str_p[count] = str_p[count].replace('</p>','')
        str_p[count] = ' '.join(str_p[count].split())
        count=count+1

    return str_p


def acs_rsc_reader(fname_in, reader):
    """call chemdataextractor reader"""
    with open('{}.html'.format(fname_in[:-5]), 'rb') as f:
        doc = Document.from_file(f, readers=reader)

    return [para.text for para in doc.paragraphs]


def generic_reader(fname_in):
    with open('{}.html'.format(fname_in[:-5]), 'rb') as f:
        doc = Document.from_file(f)

    return [para.text for para in doc.paragraphs]


def html_to_txt(fname_in, fname_out=None, doi=None, publisher='unknow', verbose=0):
    """
    convert html format paper into txt format
    
    $Param:
        fname_in - str, path to html file
        fname_out - str, path to output txt file
        doi - str, doi of paper, if not present, then try extract from filename
        publisher - str, pulisher of paper, if not present, then try get from doi
        verbose - int, 0 for no output, 1 for print success info
    
    $Return:
        int, 1 if success, 0 if failed
    """
    if not fname_out:
        fname_out = '{}.txt'.format(fname_in[:-5])
        
    if publisher == 'unknow':
        if not doi:
            doi = os.path.basename(fname_in[:-5]).replace('_', '/')
        publisher = get_publisher(doi)

    try:
        if publisher == 'acs':
            paragraphs = acs_rsc_reader(fname_in, [acs.AcsHtmlReader()])
            
        elif publisher == 'rsc':
            paragraphs = acs_rsc_reader(fname_in, [rsc.RscHtmlReader()])
            
        elif publisher == 'wiley':
            paragraphs = wiley_reader(fname_in)
        
        else:
            paragraphs = generic_reader(fname_in)    
            
        with open(fname_out, 'w') as f:
            for para in paragraphs:
                f.write('<p>')
                f.write(para.strip('\n'))
                f.write('</p>')
                f.write('\n')
        if verbose == '1':            
    		logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
        return 1
    
    except BaseException as e:
        logging.error('failed. filename: {}, doi: {}, publisher: {}, error: {}'.format(fname_in, doi, publisher, e))
        return 0


def pdf_to_txt(fname_in, fname_out=None, doi=None, publisher='unknow', verbose=0):
    """
    convert pdf format paper into txt format
    
    $Param:
        fname_in - str, path to html file
        fname_out - str, path to output txt file
        doi - str, doi of paper, if not present, then try extract from filename
        publisher - str, pulisher of paper, if not present, then try get from doi
        verbose - int, 0 for no output, 1 for print success info
    
    $Return:
        int, 1 if success, 0 if failed
    """
    if not fname_out:
        fname_out = '{}.txt'.format(fname_in[:-5])
        
    if publisher == 'unknow':
        if not doi:
            doi = os.path.basename(fname_in[:-5]).replace('_', '/').strip('SI_')
        publisher = get_publisher(doi)

    if fname_in.endswith('.pdf'):
        try:
            logging.critical('processing:{}'.format(file))
            fp = open(fname_in, 'rb')
            parser = PDFParser(fp)
            document = PDFDocument(parser)
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed
            else:
                rsrcmgr=PDFResourceManager()
                laparams=LAParams()
                device=PDFPageAggregator(rsrcmgr,laparams=laparams)
                interpreter=PDFPageInterpreter(rsrcmgr,device)
                for page in PDFPage.create_pages(document):
                    interpreter.process_page(page)
                    layout=device.get_result()
                    for x in layout:
                        if(isinstance(x,LTTextBoxHorizontal)):
                            fname_out = fname_in.replace("allsis","allsis_txt")
                            with open(fname_out.replace('.pdf', '.txt'),'ab') as f:
                                f.write(x.get_text().encode('utf-8')+b'\n')
        	if verbose == '1':            
    			logging.info('success. doi: {}, publisher: {}'.format(doi, mode))
        	return 1
	
	    except BaseException as e:
	        logging.error('failed. filename: {}, doi: {}, publisher: {}, error: {}'.format(fname_in, doi, publisher, e))
	        return 0


def line_clean(line):
    """ 
    replace wrong punction
    
    $param:
        line - string of paragraphs
    
    $return:
        line - string of paragraphs
    """
    # tab space to normal space
    line = line.replace('\t', ' ')
    
    # special qoute mark replace
    line = line.replace('′', '\'')
    line = line.replace('‘', '\'')
    line = line.replace('’', '\'')
    line = line.replace('″', '"')
    line = line.replace('“', '"')
    line = line.replace('”', '"')
    
    # invalid character to space
    line = line.replace('\ufeff', '')
    line = line.replace('\u3000', ' ')
    line = line.replace('\uf071', ' ')
    line = line.replace('\xa0', ' ')
    line = line.replace('\uf064', ' ')
    line = line.replace('\uf06d', ' ')
    line = line.replace('\u0b0C', ' ')
    line = line.replace('\u2009', ' ')
    
    # middle dot replace
    line = line.replace('\uf09e', '\u00b7')
    line = line.replace('\u2022', '\u00b7')
    line = line.replace('\u0387', '\u00b7')
    line = line.replace('\u2219', '\u00b7')
    line = line.replace('\ua78f', '\u00b7')
    line = line.replace('\uf0d7', '\u00b7')
    
    # special ~ 
    line = line.replace('∼', '~')
    
    # special '
    line = line.replace("`", "'")
    
    # micro- miu to u
    line = line.replace('\uf06d', 'u')
    line = line.replace('\u00b5', 'u')
    line = line.replace('\u03bc', 'u')
    
    # special σ
    line = line.replace('\uf073', '\u03c3')
    
    # special -
    line = line.replace('\u2212', '-')
    line = line.replace('\u2013', '-')
    
    # °C replace
    line = line.replace('\uf0b0', '°C')
    line = line.replace('\u2070C', '°C')
    line = line.replace('\u02daC', '°C')
    line = line.replace('\u2010', '-')
    line = re.sub(r"(\d{1,3}) s?0C", "\g<1> °C", line)
    line = re.sub(r"(\d{1,3})\s?oC", "\g<1> °C", line)
    line = re.sub(r"(\d{1,3})\s?ºC", "\g<1> °C", line)
    line = re.sub(r"(\d{1,3})\s?Cº", "\g<1> °C", line)
    
    # remove mutiple space
    line = re.sub(r' {2,}', ' ', line)

    return line


def file_clean(fname):
    content = []
    with open(fname, 'r') as f1:
        line = f1.readline()
        while line:
            line = line_clean(line)
            content.append(line)
            line = f1.readline()
    
    with open(fname, 'w') as f2:
        f2.writelines(content)
        
    logging.info('cleaned. fname: {}'.format(fname))
    
    return fname


def text(line):
    if len(line) <= 2:
        return True
    else:
        line = line.strip('\n')
        while line[-1] == ' ':
            if len(line) <= 1:
                return True
            line = line[:-1]
        if line[-1] == '.':
            return True
        else:
            return False


def pdf_reader(fname):
    with open(fname, 'r', errors='ignore') as f:
        content = f.readlines()
    new_content = []
    i = 0
    temp = ''
    while i < len(content):

        if text(content[i]):
            temp += content[i]
            new_content.append(temp)
            i += 1
            temp = ''
        else:
            temp+=content[i].strip('\n')
            i+=1

    new_cleaned_content = []
    for line in new_content:
        if len(line) <= 3:
            continue
        line = line_clean(line)
        line = ''.join(['<p>',line.strip(), '</p>'])
        new_cleaned_content.append(line + '\n')

    with open(fname, 'w', errors='ignore') as f:
        f.writelines(new_cleaned_content)
        
    logging.info('cleaned. fname: {}'.format(fname))
    
    return fname


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
        
    logging.info('fname: {}'.format(fname))
    logging.info('abbrs_filterd:\n{}'.format('\n'.join([str(i) for i in abbrs_filtered])))
    
    return abbrs_filtered
