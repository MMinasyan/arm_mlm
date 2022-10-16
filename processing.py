import json
import os
import numpy as np


def load_dicts():
    if 'char2idx_latarm.json' not in os.listdir('data/') and 'idx2char_latarm.json' not in os.listdir('data/'):
        armchars = 'աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքևօֆ'
        latinchars = 'abcdefghijklmnopqrstuvwxyz'
        otherchars = '1234567890,./;\'\[\]\\-=`~!@#$%^&*()_+|{}:\"<>?«»,․/՞՝՜;՛։֊―\n '
        allchars = list(set(list(armchars) + list(armchars.upper()) + list(latinchars) + list(latinchars.upper()) + list(otherchars)))

        char2idx = {'oov': 1}
        for i in range(2, len(allchars)+2):
            char2idx[allchars[i-2]] = i
        char2idx['MASK'] = len(char2idx)+1
        idx2char = {}
        for k in char2idx.keys():
            idx2char[char2idx[k]] = k

        with open('data/char2idx_latarm.json', mode='w', encoding='utf-8') as fp:
            json.dump(char2idx, fp)
        with open('data/idx2char_latarm.json', mode='w', encoding='utf-8') as fp:
            json.dump(idx2char, fp)
        print('\nfiles not existing. created new files.\nWarning: an older trained model might not work with new character indexing')
    
    with open('data/char2idx_latarm.json', mode='r', encoding='utf-8') as file:
        char2idx = json.load(file)
    with open('data/idx2char_latarm.json', mode='r', encoding='utf-8') as file:
        idx2char = json.load(file)
    return char2idx, idx2char


def encode_char(seq, vocab):
    vocab_set = set(list(vocab.keys())[1:])
    encoded_seq = np.array([int(vocab[l]) if l in vocab_set else vocab['oov'] for l in seq])
    return encoded_seq

def count_samples(filepath):
    nl = 0
    with open(filepath, mode='r', encoding='utf8') as file:
        for line in file:
            nl += 1
    return nl
