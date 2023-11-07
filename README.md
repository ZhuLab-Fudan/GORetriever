# GORetrieval

GORetrieval: A Novel Two-Stage Literature Based Deep Information Retrieval Framework for Accurate Protein Function Annotation

# Requirements

```
conda create -n goretrieval python=3.8

conda activate goretrieval

pip install -r requirements.txt
```

# inference

```
python predict.py --task bp --pro 3 --gpu 0
```

# File Structure

├── cross_model - Download

└────── *_PubMedBERT_epoch{}

└────── *_PubMedBERT_epoch1

├── data - training and testing data

└────── golden - evaluate files

└────── *_trans.txt - train set

└────── test.txt - test set

├── file

└────── bp_pro2go.npy - proteinid -> GO list

└────── pmid2text.npy - Pubmed ID -> title and abstract 

└────── proid2name.npy - proteinid -> protein name


└────── *_dev_t5_texts.npy - extracted sentences

└────── *_retrieval_all.npy

└────── *_retrieval_pro_3.npy - retrieval result

└────── *_t5_scores.npy - score cache


├── go_index
  
├── predict.py

├── pro_index

├── result - final result

├── test
