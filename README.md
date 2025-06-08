# GORetriever

Official code for [GORetriever: Reranking protein-description-based GO candidates by literature-driven deep information retrieval for protein-function-annotation](https://academic.oup.com/bioinformatics/article/40/Supplement_2/ii53/7749084)

## News

- **2025.06.08**  
  We have released large-scale predictions using **GORetriever** on all 420,000 **Swiss-Prot** proteins.
  
  👉 [Results available here](https://drive.google.com/file/d/1FoKshfeQ_JeHCfRJWaIjjL-GG586Tz1B/view?usp=sharing)
  
  👉 [For more details](https://github.com/ZhuLab-Fudan/GORetriever/tree/main/result)

- **2025.04.07**  
  Our follow-up work **GOAnnotator: Accurate Protein Function Annotation Using Automatically Retrieved Literature** has been accepted to **ISMB/ECCB 2025**!  
  👉 Code available at [GOAnnotator GitHub](https://github.com/ZhuLab-Fudan/GOAnnotator)

- **2024.05.30**  
  Our paper on **GORetriever** has been accepted to **ECCB 2024**!

# Requirements

python=3.8

sentence-transformers == 2.2.2

pyserini == 0.10.1.0

pygaggle == 0.0.3.1

faiss-cpu == 1.7.4

numpy == 1.23.5

nltk == 3.8.1

# Inference

Download cross_encoder from https://drive.google.com/file/d/11W51FnM62Z79qGPkuZHRzAv6Bx_L1Mah/view?usp=sharing into folder **cross_model**

```
python predict.py \
  --task [branch] \
  --pro [k] \
  --gpu [n] \
  --file [file]
```

branch: bp, mf, cc

pro: number of retrieved proteins, set k = 3 for MFO and BPO and k = 2 for CCO 

gpu: cuda number, default as 0

file: filename for test set, eg: ./data/test.txt

eg:
```
python predict.py --task bp --pro 3 --gpu 0
```

# Start your own experiments

If you want to start your new experiment, delete the files in folders **test, file, pro_index, go_index**. Otherwise the program will reason based on these pre-existing files.

Rewrite:

- *_pro2go.npy - proteinid -> GO list dictionary in the training data
- pmid2text.npy - Pubmed ID -> title and abstract dictionary for the PubMed article you will use in the training and testing data
- proid2name.npy - proteinid -> protein name dictionary for the PubMed article you will use in the training and testing data
- go_index, pro_index - index built by pyserini

Delete (will be rebuilt after running predict.py):

- *_dev_t5_texts.npy - extracted sentences
- *_retrieval_all.npy
- *_retrieval_pro_3.npy - retrieval result
- *_t5_scores.npy - score cache

# File Structure

├── cross_model - Download

└────── *_PubMedBERT_epoch{}

└────── *_PubMedBERT_epoch1

├── data - training and testing data

└────── golden - evaluate files

└────── *_train.txt - train set

└────── test.txt - test set

├── file

└────── *_pro2go.npy - proteinid -> GO list

└────── pmid2text.npy - Pubmed ID -> title and abstract 

└────── proid2name.npy - proteinid -> protein name

├── test

└────── *_dev_t5_texts.npy - extracted sentences

└────── *_retrieval_all.npy

└────── *_retrieval_pro_3.npy - retrieval result

└────── *_t5_scores.npy - score cache

├── go_index

├── pro_index

├── result - final result

├── predict.py

└── data_preprocess.py
