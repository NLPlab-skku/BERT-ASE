import pdb
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

import time
import datetime


def load_file(path):
    with open(path) as f:
        data = f.readlines()
    return data


def tokenizing_neutral(neutral_list, tokenizer):

    tokenized_list = []
    for word in neutral_list:
        tmp = tokenizer.encode(word, add_special_tokens=False)
        tokenized_list.append(tmp)

    return tokenized_list


def calculate_gender_vector(wordlist, tokenizer, model):
    model.eval()
    # make words to sequence
    female_seq = " ".join(wordlist["female"])
    male_seq = " ".join(wordlist["male"])

    female_encoded = tokenizer(female_seq, return_tensors="pt").to("cuda")
    male_encoded = tokenizer(male_seq, return_tensors="pt").to("cuda")

    female_output = model(**female_encoded)
    male_output = model(**male_encoded)

    female_vec = torch.sum(female_output.hidden_states[-1].squeeze(dim=0)[1:-1], dim=0) # [1:-1] : except CLS & SEP
    male_vec = torch.sum(male_output.hidden_states[-1].squeeze(dim=0)[1:-1], dim=0)

    gender_vec = (male_vec - female_vec) / len(wordlist["male"])

    return gender_vec.to("cuda")


def make_debias_id(input_ids, neutral_tok):
    debias_ids = [0] * len(input_ids)

    for neu in neutral_tok:
        if neu[0] in input_ids:
            start_pos = input_ids.index(neu[0])

            if neu == input_ids[start_pos:start_pos+len(neu)]:
                for i in range(start_pos, start_pos+len(neu)):
                    debias_ids[i] = 1

    return debias_ids


def convert_examples_to_features(tokenizer, sentences, labels, neutral_tok, args=None):
    print(' Original: ', sentences[0])
    print('Tokenized: ', tokenizer.tokenize(sentences[0]))
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

    input_ids = []
    attention_masks = []
    masked_lm_labels = []
    debiasing_labels = [] # haeun

    MAX_LEN = 164

    for sentence, label in zip(sentences, labels):
        if args.model_name == "roberta-base":
            sentence = sentence.replace("[CLS]", tokenizer.bos_token)
            sentence = sentence.replace("[SEP]", tokenizer.sep_token)
            sentence = sentence.replace("[MASK]", tokenizer.mask_token)
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        sentence_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))[:MAX_LEN]
        encoded_sent = tokenizer.encode(sentence, add_special_tokens=False)[:MAX_LEN]
        att_mask = [1] * len(encoded_sent)
        label_id = tokenizer.convert_tokens_to_ids(label)
        label_id = [label_id if id == mask_id else -100 for id in sentence_ids]

        # make debias_id - only contains idx
        debias_ids = make_debias_id(sentence_ids, neutral_tok)

        padding_length = MAX_LEN - len(encoded_sent) if len(encoded_sent) < MAX_LEN else 0

        debias_padding = MAX_LEN - len(debias_ids) if len(debias_ids) < MAX_LEN else 0

        encoded_sent += [0] * padding_length
        att_mask += [0] * padding_length
        label_id += [-100] * padding_length
        debias_ids += [0] * debias_padding

        # import pdb; pdb.set_trace()

        assert len(encoded_sent) == len(label_id) == len(att_mask) == len(debias_ids)

        input_ids.append(encoded_sent)
        attention_masks.append(att_mask)
        masked_lm_labels.append(label_id)
        debiasing_labels.append(debias_ids)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    return (input_ids, attention_masks, masked_lm_labels, debiasing_labels)


def convert_features_to_dataset(features):
    # import pdb; pdb.set_trace()
    all_input_ids = torch.tensor(features[0], dtype=torch.long)
    all_attention_masks = torch.tensor(features[1], dtype=torch.long)
    all_masked_lm_ids = torch.tensor(features[2], dtype=torch.long)
    all_debiasing_labels = torch.tensor(features[3], dtype=torch.long)

    return TensorDataset(all_input_ids, all_attention_masks, all_masked_lm_ids, all_debiasing_labels)


def load_data(mode="augmented"):
    # if not os.path.exists("train.csv"):
    if mode == "augmented":
        df_orig = pd.read_csv('./data/original_data.csv')
        df_flipped = pd.read_csv('./data/flipped_data.csv')
        df = pd.concat([df_orig, df_flipped])
        df['gender'] = df['pronouns'].str.contains('^he$|^his$|^him$').astype(int)
    elif mode == "unaugmented":
        df = pd.read_csv('./data/original_data.csv')
        df['gender'] = df['pronouns'].str.contains('^he$|^his$|^him$').astype(int)

    return df
    # return (train, val)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    labels_flat_filtered = (labels_flat != 0) * (labels_flat != -100) * labels_flat

    return np.sum((pred_flat == labels_flat_filtered) * (labels_flat_filtered != 0)) / sum(labels_flat_filtered != 0)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def load_stereo(path):
    data = pd.read_csv(path, sep="\t", header=None)
    neutral = data[0].values.tolist()
    neutral += data[1].values.tolist()

    return neutral


