import numpy as np
from docopt import docopt
import torch
from transformers import *
import pickle
from tqdm import tqdm

import pdb

def read_data_file(input_file):
 
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_lm():

    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-multilingual-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    return model, tokenizer


def tokenize(tokenizer, data):
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['hard_text'], add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


def encode_text(model, data):
    """
    encode the text
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    batch = []
    model.cuda()
    for row in tqdm(data):

        batch.append(row)
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids.cuda())[0]
            all_data_avg.append(last_hidden_states.cpu().squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.cpu().squeeze(0)[0].numpy())
        batch = []


    return np.array(all_data_avg), np.array(all_data_cls)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_file = arguments['--input_file']
    out_dir = arguments['--output_dir']
    split = arguments['--split']


    model, tokenizer = load_lm()

    data = read_data_file(in_file)
    tokens = tokenize(tokenizer, data)

    avg_data, cls_data = encode_text(model, tokens)

    np.save(out_dir + '/' + split + '_avg_mbert.npy', avg_data)
    np.save(out_dir + '/' + split + '_cls_mbert.npy', cls_data)

