'''Анализатор настроения украинского текста со списками позитивных/негативных слов
проекта Chen-Skiena для украинского языка, отредактированными мной.
Берет по одному предложению из текста, оценивает как позитивное или негативное,
добавляет в словарь предложение:оценка.
Затем словарь преобразуется в датасет
'''

import pandas as pd
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='uk')

def norm_words(text):
    '''
Функция берет украинский текст, приводит каждое слово к нормальной форме,
вычленяет существительные, прилагательные и глаголы и выводит их в список
    '''
    relev_words = list()
    words = simple_preprocess(text)
    for word in words:
        p = morph.parse(word)[0]
#        if 'NOUN' in p.tag or 'VERB' in p.tag or 'ADJF' in p.tag:
        relev_words.append(p.normal_form)
    return relev_words

with open('negative_words_uk_ed.txt', 'r', encoding='utf-8') as n, \
     open('positive_words_uk_ed.txt', 'r', encoding='utf-8') as p, \
     open('Тіні забутих предків.txt', 'r', encoding='utf-8') as t:
    textN = n.read()
    textP = p.read()
    list_n=textN.split()
    list_p=textP.split()
    txt=t.read()
'''
txt='Це була чудова ніч. Але мій настрій був нейтральний. Декілька поганих людей трошки зіпсували його'
'''
snts=sent_tokenize(txt)
dct={}
for ele in snts:
    snt_score=0
    w=norm_words(ele)
    for el in w:
        if el in list_p:
            snt_score+=1
        elif el in list_n:
            snt_score-=1
    if snt_score>0:
        fin_score=1
    elif snt_score<0:
        fin_score=-1
    else:
        fin_score=0
    dct[ele]=fin_score

df = pd.DataFrame.from_dict(dct, orient="index")
# orient='index' to create the DataFrame using dictionary keys as rows
df.to_csv("dataset.csv")

#print(df)
