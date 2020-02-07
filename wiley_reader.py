from bs4 import BeautifulSoup

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