import json
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

def data_generator(filepath='data/all_paragraphs.txt', max_len=512, masking=True, batch_size=64, vocab=None, mask_id=None):
    char_set = set(vocab.keys())
    if not mask_id:
        mask_id = len(vocab)
    while True:
        try:
            with open(filepath, encoding='utf8', mode='r') as file:
                data_batch = []
                for line in file:
                    if len(data_batch) == batch_size:
                        y_labels = []
                        weights_par = []
                        masked_par = []
                        for p in data_batch:
                            char_seq = np.array(list(p))
                            len_par = len(char_seq)
                            y_label = np.array([vocab[c] if c in char_set else 1 for c in char_seq])
                            y_labels.append(y_label)
                            randidx = np.random.rand(len_par) <= 0.15
                            randidx[y_label==1] = False
                            weights_seq = np.zeros((len_par, ))
                            weights_seq[randidx] = 1
                            weights_seq = weights_seq.astype('int16')
                            #weights_seq[token_seq=='oov'] = 0
                            masked_seq = encode_char(p, vocab)
                            masked_seq[randidx] = mask_id
                            weights_par.append(weights_seq)
                            masked_par.append(masked_seq)
                        y_labels = pad_sequences(y_labels, maxlen=max_len, padding='post')
                        weights_par = pad_sequences(weights_par, maxlen=max_len, padding='post')
                        masked_par = pad_sequences(masked_par, maxlen=max_len, padding='post')
                        yield masked_par, y_labels, weights_par
                        data_batch = []
                    else:
                        data_batch.append(line)
        except StopIteration:
            pass