import sys
import torch
import kaldiio

import os

path = os.getcwd()

arguments = sys.argv[1:]
if arguments != []:
    argument = arguments[0] 

train_mfcc = './feat/mfcc_train_39.ark'
val_mfcc = './feat/mfcc_val_39.ark'
train_spec = './feat/spec_train.ark'
val_spec = './feat/spec_val.ark'
train_fbank = './feat/fbank_train.ark'
val_fbank = './feat/fbank_val.ark'
def get_max(ark_path):
    feats_dict = kaldiio.load_ark(ark_path)
    feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
    length_max = 0
    for feat_label in feats_list:
        length = feat_label[0].shape[0]
        d_num = feat_label[0].shape[1]
        if length > length_max: 
            length_max = length
    return length_max, d_num
def check(ark_path):
    error = []
    feats_dict = kaldiio.load_ark(ark_path)
    name_list = [uttid for uttid, feat in feats_dict]
    for name in name_list:
        n1 = int(name.split('-')[1])
        n2 = int(name.split('-')[2])
        n3 = int(name.split('-')[3])
        if n1<4 or n2<4 or n3<4:
            error.append(name)
    return error

if __name__ == '__main__':
    if argument == 'mfcc' or argument is None or argument == 'all':
        l_train = get_max(train_mfcc)
        l_val = get_max(val_mfcc)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'MFCC: seq_length is {max}, n_features is {n_features}.')
    if argument == 'spec' or argument == 'all':
        l_train = get_max(train_spec)
        l_val = get_max(val_spec)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'SPEC: seq_length is {max}, n_features is {n_features}.')
    if argument == 'fbank' or argument == 'all':
        l_train = get_max(train_fbank)
        l_val = get_max(val_fbank)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'SPEC: seq_length is {max}, n_features is {n_features}.')