import os

import random
import argparse
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--task", type=str, default='cc')
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

print('--gpu', args.gpu)
print('--model', args.model)
print('--epochs', args.epochs)
print('--batch_size', args.batch_size)

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

from torch.utils.data import DataLoader
import math
from sentence_transformers import CrossEncoder
from sentence_transformers import InputExample
import json


# Read the dataset
def loadJsonData(data_path):
    assert type(data_path) == type([])
    json_data = []
    for path in data_path:
        f = open(path, 'r')
        json_data.extend(json.load(f))
    return json_data


def trainModel(model, train_pos_data, train_neg_data, num_epochs, train_batch_size, model_save_path):
    random.seed(123)
    train_samples = []
    dev_samples = []
    random.shuffle(train_pos_data)
    random.shuffle(train_neg_data)
    split_dev_test(train_samples, dev_samples, train_pos_data, train_neg_data, percentage=0.9)
    random.shuffle(train_samples)
    random.shuffle(dev_samples)
    print("batch size, ", train_batch_size)
    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size * len(device_ids))
    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='sts-dev')
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    # logger.info("Warmup-steps: {}".format(warmup_steps))
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              use_amp=True,
              device_ids = device_ids)  


def start_train_get_model(num_epochs, train_batch_size, start_epoch=0):
    random.seed(123)
    trainDocNewFile_pos = [
        f'./train/{args.task}_t5_all_pos.json'
    ]

    trainDocNewFile_neg = [
        f'./train/{args.task}_t5_all_neg.json'
    ]
 

    if len(trainDocNewFile_neg) != len(trainDocNewFile_pos):
        data_all = loadJsonData(trainDocNewFile_pos)
        data_all.extend(loadJsonData(trainDocNewFile_neg))
        train_pos_data = []
        train_neg_data = []
        for item in data_all:
            if item['label'] == 0:
                train_neg_data.append(item)
            else:
                train_pos_data.append(item)
    else:
        train_pos_data = loadJsonData(trainDocNewFile_pos)
        train_neg_data = loadJsonData(trainDocNewFile_neg)

    print("pos:", len(train_pos_data), "neg:", len(train_neg_data))
    # Define our Cross-Encoder
    # We use microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract as base model and set num_labels=1,
    # which predicts a continous score between 0 and 1

    model_path = 'PubMedBERT_epoch{}'
    
    if start_epoch == 0:

        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        model = CrossEncoder(model_name, num_labels=1, max_length=512)  
    model_save_path = f'./cross_model/{args.task}_' + model_path


    print("Save Path: ", model_save_path)
    trainModel(model, train_pos_data, train_neg_data, num_epochs, train_batch_size, model_save_path)


def get_model_path(model_save_path):
    model = CrossEncoder(model_save_path, max_length=512) 
    return model
  

def split_dev_test(dev, test, pos_train, neg_train, percentage=0.9):
    max_len = int(len(pos_train) * percentage)
    test_pos = []
    test_neg = []
    for obj in pos_train[:max_len]:
        dev.append(InputExample(texts=[obj['query'], obj['contents']],
                                label=obj['label']))

    for obj in pos_train[max_len:]:
        test_pos.append(InputExample(texts=[obj['query'], obj['contents']],
                                 label=obj['label']))

    for obj in neg_train[:max_len]:
        dev.append(InputExample(texts=[obj['query'], obj['contents']],
                                label=obj['label']))
    for obj in neg_train[max_len:]:
        test_neg.append(InputExample(texts=[obj['query'], obj['contents']],
                                 label=obj['label']))
    test_pos.extend(test_neg)
    test.extend(test_pos)
    random.shuffle(test)
    return dev, test


if __name__ == '__main__':
    start_train_get_model(args.epochs, args.batch_size, start_epoch=0)

# nohup python -u train.py --task cc > ./log/train.log 2>&1 &
