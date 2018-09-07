'''
This code generate word embedding for each of the words in vocabulary and stores them in .pkl format

Sumana Basu
260727568
'''

import cPickle as pickle
from collections import Counter
import os
import numpy as np
from itertools import chain

empty = 0
eos = 1
start_idx = eos + 1
vocab_size = 40000
embedding_dim = 100
glove_n_symbols = 400000
glove_index_dict = {}
glove_thr = 0.5
word2glove = {}

# Load Glove embeddings
# Should be stored in .keras/datasets/
fname = 'glove.6B.%dd.txt' % embedding_dim
datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
datadir = os.path.join(datadir_base, 'datasets')
glove_name = os.path.join(datadir, fname)


def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount


def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx, word) for word, idx in word2idx.iteritems())
    return word2idx, idx2word


def get_glove_embedding_weights():
    glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
    globale_scale=.1
    with open(glove_name, 'r') as fp:
        i = 0
        for l in fp:
            l = l.strip().split()
            w = l[0]
            glove_index_dict[w] = i
            glove_embedding_weights[i,:] = map(float,l[1:])
            i += 1
    glove_embedding_weights *= globale_scale
    return glove_embedding_weights


def calc_glove_index_dict():
    for w,i in glove_index_dict.iteritems():
        w = w.lower()
        if w not in glove_index_dict:
            glove_index_dict[w] = i


def wordEmbed(idx2word):
    shape = (vocab_size, embedding_dim)
    glove_embedding_weights = get_glove_embedding_weights()
    scale = glove_embedding_weights.std()*np.sqrt(12)/2
    embedding = np.random.uniform(low=-scale, high=scale, size=shape)
    c = 0
    for i in range(vocab_size):
        w = idx2word[i]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is None and w.startswith('#'):
            w = w[1:]
            g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is not None:
            embedding[i,:] = glove_embedding_weights[g,:]
            c+=1
    return  embedding


def convWord2Glove(word2idx):
    for w in word2idx:
        if w in glove_index_dict:
            g = w
        elif w.lower() in glove_index_dict:
            g = w.lower()
        elif w.startswith('#') and w[1:] in glove_index_dict:
            g = w[1:]
        elif w.startswith('#') and w[1:].lower() in glove_index_dict:
            g = w[1:].lower()
        else:
            continue
        word2glove[w] = g


def oov(embedding, word2idx, idx2word, glove_embedding_weights):
    normed_embedding = embedding / np.array([np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]

    nb_unknown_words = 100

    glove_match = []
    for w, idx in word2idx.iteritems():
        if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
            gidx = glove_index_dict[word2glove[w]]
            gweight = glove_embedding_weights[gidx, :].copy()
            # find row in embedding that has the highest cos score with gweight
            gweight /= np.sqrt(np.dot(gweight, gweight))
            score = np.dot(normed_embedding[:vocab_size - nb_unknown_words], gweight)
            while True:
                embedding_idx = score.argmax()
                s = score[embedding_idx]
                if s < glove_thr:
                    break
                if idx2word[embedding_idx] in word2glove:
                    glove_match.append((w, embedding_idx, s))
                    break
                score[embedding_idx] = -1
    glove_match.sort(key=lambda x: -x[2])

    return  glove_match


def main():
    with open('hindu_data.pkl', 'rb') as fp:
        heads, desc, keywords = pickle.load(fp)

    heads = [h.lower() for h in heads]
    desc = [h.lower() for h in desc]
    vocab, vocabcount = get_vocab(heads + desc)
    word2idx, idx2word = get_idx(vocab, vocabcount)

    calc_glove_index_dict()
    embedding = wordEmbed(idx2word)
    convWord2Glove(word2idx)
    glove_embedding_weights = get_glove_embedding_weights()
    glove_match = oov(embedding, word2idx, idx2word, glove_embedding_weights)

    glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)

    X = [[word2idx[token] for token in d.split()] for d in desc]
    Y = [[word2idx[token] for token in headline.split()] for headline in heads]

    with open('vocabulary-embedding.pkl', 'wb') as fp:
        pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, -1)

    with open('vocabulary-embedding.data.pkl', 'wb') as fp:
        pickle.dump((X, Y), fp, -1)


if __name__ == '__main__':
    main()
