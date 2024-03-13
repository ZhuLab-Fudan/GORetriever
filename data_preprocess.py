import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import random 
import nltk
from sentence_transformers import CrossEncoder
from transformers import T5ForConditionalGeneration
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import torch.nn as nn


task_def = {
    'cc': 'Cellular Component',
    'bp': 'Biological Process',
    'mf': 'Molecular Function',
}

def get_text(url): 
    headers = {} 
    user_agent_list = [
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) Gecko/20100101 Firefox/61.0",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
]
    headers['User-Agent'] = random.choice(user_agent_list)

    content = requests.get(url, headers = headers).text
    # time.sleep(10)
    soup = bs(content, 'lxml')
    # text = ''
    try:
        title = soup.select('#full-view-heading > h1')[0].get_text().replace('\n', '').replace('\t', '').strip()
    except IndexError:
        # print(url)
        text = ''
    else:
        try:
            abstracts = soup.select('#eng-abstract > p')
            abstract = ''
            for item in abstracts:
                abstract = abstract + item.get_text().replace('\n', '').replace('\t', '').strip()
            # abstract = soup.select('#eng-abstract > p')[0].get_text().replace('\n', '').strip()
            text = title+'.'+abstract
        except IndexError:
            text=''

    return text



def data_t5_extract(task):
    '''
    ertract informative sentences
    '''
    print('data extract...')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-med-msmarco')
    model = nn.DataParallel(model)
    model = model.cuda().module
    T5tokenizer = MonoT5.get_tokenizer('t5-base')
    reranker = MonoT5(model, T5tokenizer)
    proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
    pro2text = {}
    pmid2text = np.load('./file/pmid2text.npy', allow_pickle=True).item()
    with open(f'./data/{task}_trans.txt') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.split('\t')
            pro = line[0]
            pmid = line[2].replace('\n', '')
            try:
                proname = proid2name[pro]
            except KeyError:
                continue

            if not pro2text.get(pro):
                pro2text[pro] = []

            text = pmid2text[pmid]
            
            sentences = tokenizer.tokenize(text)
            if len(sentences) == 1:
                continue
            if sentences[0] in pro2text[pro]:
                continue
            pro2text[pro].extend(sentences)

        for pro in tqdm(pro2text): 
            query = f"what is the {task_def[task]} of protein {proname}?"
            sentences = pro2text[pro]
            if len(sentences) < 3:
                print(sentences)
                continue
            texts = []
            for sentence in sentences:
                texts.append(Text(sentence, {}, 0))
            scores = reranker.rerank(Query(query), texts)
            reranked = {}
            for i in range(len(scores)):
                reranked[sentences[i]] = float(scores[i].score)
            reranked = sorted(reranked, key = lambda x:reranked[x], reverse=True)
            res = reranked[:len(reranked)//2]
            pro2text[pro] = ' '.join(res)

    np.save(f'./train/{task}_t5_texts.npy', pro2text)
    
    return pro2text

def build_neg(task):
    '''
    bm25 retrieval
    return go list
    '''
    searcher = LuceneSearcher('./pro_index/')
    with open(f'./data/{task}_trans.txt') as f:
        pro2go = {}
        lines = f.readlines()
        for line in lines:
            l = line.split('\t')
            pro2go[l[0]] = pro2go.get(l[0], []) + [l[1]]
    with open(f'./data/{task}_trans.txt') as f:
        lines = f.readlines()
        pro2doc = {}
        pro2sample = {}
        for line in lines:
            line = line.split('\t')
            proid = line[0]
            pro2sample[proid] = pro2sample.get(proid, 0) + 1
            if not pro2doc.get(proid):
                pro2doc[proid] = line[2]
                
    wf = open(f'./train/{task}_all_neg.txt', 'w')
    for id in tqdm(pro2doc):
        res = []
        try:
            text = searcher.doc(id).raw()
        except AttributeError:
            continue

        l = searcher.search(text, 100)
        for _ in l:
            _ = json.loads(_.raw)
            if _['id'] == id:
                continue
            else:
                if pro2go.get(_['id']):
                    for go in pro2go[_['id']]:
                        if not go in res and not go in pro2go[id]:
                            res.append(go)
                
                # res.extend(pro2go.get(_['id'], []))
        negs = []
        for go in res:
            negs.append(id + '\t' + go + '\t' + pro2doc[id])

        negs = random.sample(negs, pro2sample[id] if len(negs) > pro2sample[id] else len(negs))
        wf.write(''.join(negs))

def text2trainjson(task):
    '''
    build trainjson for cross-encoder
    '''
    pro2text = np.load(f'./train/{task}_t5_texts.npy', allow_pickle=True).item()
    proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
    goreader = LuceneSearcher('./go_index/')

    with open(f'./train/{task}_all_neg.txt') as f:
        data = []
        with open(f'./train/{task}_t5_all_neg.json', 'w') as wf:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split('\t')
                proid = line[0]
                goid = line[1]
                try:
                    contents = json.loads(goreader.doc(goid.replace("GO:", '').replace("\n", "")).raw())['contents']
                except AttributeError:
                    print(goid)
                    continue
                data.append({
                    'query': f"The protein is \"{proid2name[proid]}\", the document is \"{pro2text[proid]}\".",
                    'contents': contents,
                    'label': 0
                })
            json.dump(data, wf, indent=2)
    
    with open(f'./data/{task}_trans.txt') as f:
        data = []
        with open(f'./train/{task}_t5_all_pos.json', 'w') as wf:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split('\t')
                proid = line[0]
                goid = line[1]
                try:
                    proname = proid2name[proid]
                except KeyError:
                    continue
                try:
                    contents = json.loads(goreader.doc(goid.replace("GO:", '').replace("\n", "")).raw())['contents']
                except AttributeError:
                    print(goid)
                    continue
                data.append({
                    'query': f"The protein is \"{proname}\", the document is \"{pro2text[proid]}\".",
                    'contents': contents,
                    'label': 1
                })
            json.dump(data, wf, indent=2)    

    
if __name__ == '__main__':
    for task in ['cc', 'mf', 'bp']:
        text2trainjson(task)