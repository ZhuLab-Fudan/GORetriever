import os
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpu", type=str, default='1')
parser.add_argument("--task", type=str, default='cc')
parser.add_argument("--pro", type=str, default='0')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

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


def data_extract(args):
    '''
    sentence extract
    return:
        pro2text[dictionary]:{
            'protein_id': extracted context(string)
        }
    '''
    save_file = f'./test/{args.task}_t5_dev_texts_num.npy'
    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print('load from: ', save_file)
        print('data extract end!')
        return data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-med-msmarco')
    model = nn.DataParallel(model)
    model = model.cuda().module
    T5tokenizer = MonoT5.get_tokenizer('t5-base')
    reranker = MonoT5(model, T5tokenizer)
    proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
    pmid2text = np.load('./file/pmid2text.npy', allow_pickle=True).item()
    pro2text = {}
    with open('./data/test.txt') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.split('\t')
            pro = line[0]
            pmid = line[1].replace('\n', '')
            try:
                proname = proid2name[pro]
            except KeyError:
                print('Missing Protein Name', proname)
                continue

            if not pro2text.get(pro):
                pro2text[pro] = []

            text = pmid2text[pmid] 
            sentences = tokenizer.tokenize(text)
            # Discard information from documents containing only one sentence
            if len(sentences) == 1:
                continue
            if sentences[0] in pro2text[pro]:
                continue
            pro2text[pro].extend(sentences)

    text_score = {}
    for pro in tqdm(pro2text): 
        query = f"what is the {task_def[args.task]} of protein {proname}?"
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
        text_score[pro] = reranked

        reranked = sorted(reranked, key=lambda x:reranked[x], reverse=True)
        res = reranked[:len(reranked)//2]
        pro2text[pro] = ' '.join(res)
        

    np.save(save_file.replace('texts', 'texts_scores'), text_score)
    np.save(save_file, pro2text)
    print('data extract end!')
    return pro2text


def all_retrieval_dict(args):
    save_file = f'./test/{args.task}_retrieval_all.npy'
    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print(save_file)
        print("All retrieval end!") 
        return data
    searcher = LuceneSearcher('./pro_index/')
    pro2text = data_extract(args)
    pro2go = np.load(f'./file/{args.task}_pro2go.npy', allow_pickle=True).item()
    proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
    retrieval_dict = {}
    for proid in tqdm(pro2text):
        k = 0
        res = []
        try:
            proname = json.loads(searcher.doc(proid).raw())['contents']
        except AttributeError:
            print('Missing Protein Name', proname)
            continue
        
        l = searcher.search(proname, 3000)
        for _ in l:
            if k > int(args.pro) + 2:
                break
            _ = json.loads(_.raw)
            if _['id'] == proid:
                continue
            else:
                if pro2go.get(_['id']):
                    k += 1
                    res.append(pro2go[_['id']])
                else:
                    res.append([])
        if k < int(args.pro) + 2:
            print(k, proid)
        retrieval_dict[proid] = res
    np.save(save_file, retrieval_dict)
    print('write:', save_file)
    print("All retrieval end!") 
    return retrieval_dict      

def retrieval(args):
    save_file = f'./test/{args.task}_retrieval.npy'
    if args.pro != '0':
        save_file = save_file.replace('retrieval', f'retrieval_pro_{args.pro}')
    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print(save_file)
        print("retrieval end!") 
        return data
    retrieval_dict = all_retrieval_dict(args)
    d = {}
    for proid in tqdm(retrieval_dict):
        k = 0
        res = []
        for l in retrieval_dict[proid]:
            if k==int(args.pro):
                # print(k)
                break
            if len(l) == 0:
                continue
            else:
                k += 1
                res.extend(l)
        d[proid] = res

    np.save(save_file, d)
    print("write:", save_file)
    print("retrieval end!")
    return d

def rerank(args):
    retrieval_data = retrieval(args)
    pro2text = data_extract(args)
    score = {}
    score_dict = f'./test/{args.task}_t5_scores.npy'
    if os.path.exists(score_dict):
        print("score cache: ", score_dict)
        score = np.load(score_dict, allow_pickle=True).item()
    goreader = LuceneSearcher('./go_index/')
    proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
    model = CrossEncoder(f'./cross_model/{args.task}_PubMedBERT_epoch1/', max_length=512)
    data = []
    

    for proid in tqdm(retrieval_data):
        predicts = []
        goids = []
        try:
            proname = proid2name[proid]
        except KeyError:
            continue
        try:
            query = f"The protein is \"{proname}\", the document is \"{pro2text[proid]}\"."
        except KeyError:
            print('text', proid)
            continue
        for goid in retrieval_data[proid]:
            if len(retrieval_data[proid]) == 0:
                continue
            if not score.get(proid):
                score[proid] = {}
            else:
                if score[proid].get(goid):
                    continue
            try:
                contents = json.loads(goreader.doc(goid.replace("GO:", '').replace("\n", "")).raw())['contents']
            except AttributeError:
                continue
            else:
                goids.append(goid)
                predicts.append([query, contents])
        if len(predicts) == 0:
            continue
        scores = model.predict(predicts,  batch_size = 96, show_progress_bar=False)
        for i in range(len(scores)):
            score[proid][goids[i]] = '%.3f'%float(scores[i])
            
    np.save(score_dict, score)  

    for proid in tqdm(retrieval_data):
        if len(retrieval_data[proid]) == 0:
            print("proid", proid)
            continue
        if not score.get(proid):
            print("proid", proid)
            continue           
        res = {}
        for goid in retrieval_data[proid]:
            res[goid] = str(score[proid].get(goid, 0))
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:50]
        for item in res:
            data.append(proid + '\t' + item[0] + '\t' + item[1] + '\n')

    
    save_file = f'./result/{args.task}_t5_dev_rerank.txt'
    if args.pro != '0':
        save_file = save_file.replace('rerank', f'rerank_pro_{args.pro}')
    print(save_file)
    with open(save_file, 'w') as wf:
        wf.write(''.join(data))
    print('rerank end!')


if __name__ == '__main__':
    rerank(args)
