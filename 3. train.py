'''
This code trains the Global Attention based Stacked LSTM Model

Sumana Basu
260727568
'''

import os
import keras
from utils import log_slack
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
from context import simple_context
from keras.optimizers import Adam, RMSprop
from utils import inspect_model,lpadd,keras_rnn_predict,vocab_fold,vocab_unfold
from beamsearch import beamsearch
from generator import gensamples,predsamples
import sys
import Levenshtein
import unicodecsv as csv
from sklearn.cross_validation import train_test_split
import cPickle as pickle

#### hyperparameters and tuning variables
FN = 'train'
FN0 = 'vocabulary-embedding'
FN1 = 'train'
maxlend=100
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3
batch_norm=False
activation_rnn_size = 40 if maxlend else 0
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=32
nflips=0
nb_train_samples = 14828
nb_val_samples = 3000
########

def build_model():
    random.seed(seed)
    np.random.seed(seed)

    regularizer = l2(weight_decay) if weight_decay else None

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
                        name='embedding_1'))
    for i in range(rnn_layers):
        lstm = LSTM(rnn_size, return_sequences=True,
                    W_regularizer=regularizer, U_regularizer=regularizer,
                    b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                    name='lstm_%d'%(i+1)
                      )
        model.add(lstm)
        model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))

    if activation_rnn_size:
        model.add(SimpleContext(name='simplecontext_1'))
    model.add(TimeDistributed(Dense(vocab_size,
                                    W_regularizer=regularizer, b_regularizer=regularizer,
                                    name = 'timedistributed_1')))
    model.add(Activation('softmax', name='activation_1'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    K.set_value(model.optimizer.lr,np.float32(LR))
    inspect_model(model)
    return model

# load previously computed model weights
def load_model_weights(model):
    if FN1:
        model.load_weights('data/%s.hdf5'%FN1)
    return model

# predict sample to test working is ok
def sample_predict():
    print 'PAdding SEQUENCES'
    samples = [lpadd([3]*26)]
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    np.all(data[:,maxlend] == eos)
    data.shape,map(len, samples)
    probs = model.predict(data, verbose=0, batch_size=1)
    probs.shape

# write predictions in csv
def write_predictions():
    with open('results.csv','wb') as fp:
        writer = csv.writer(fp,delimiter=',',encoding='utf-8')
        writer.writerow(['original','predicted'])
        predsamples(X_train,Y_train,idx2word,skips=2, batch_size=batch_size, k=10, temperature=1.,use_unk=False,writer=writer)


if __name__ == '__main__':
    # load previously computed embeddings
    with open('data/%s.pkl'%FN0, 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    vocab_size, embedding_size = embedding.shape
    with open('data/%s.data.pkl'%FN0, 'rb') as fp:
        X, Y = pickle.load(fp)

    nb_unknown_words = 10

    for i in range(nb_unknown_words):
        idx2word[vocab_size-1-i] = '<%d>'%i

    oov0 = vocab_size-nb_unknown_words

    for i in range(oov0, len(idx2word)):
        idx2word[i] = idx2word[i]+'^'

    # split samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
    len(X_train), len(Y_train), len(X_test), len(Y_test)

    # set unk characters
    empty = 0
    eos = 1
    idx2word[empty] = '_'
    idx2word[eos] = '~'

    r = next(gen(X_train, Y_train, batch_size=batch_size))
    r[0].shape, r[1].shape, len(r)

    history = {}

    traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
    valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

    # start training
    r = next(traingen)
    r[0].shape, r[1].shape, len(r)

    print 'TRAINING STARTS'

    # run for 250 iterations
    for iteration in range(250):
        print 'Iteration', iteration
        log_slack('Iteration {}'.format(iteration))
        h = model.fit_generator(traingen, samples_per_epoch=nb_train_samples,
                            nb_epoch=1, validation_data=valgen, nb_val_samples=nb_val_samples
                               )
        log_slack(h.history["loss"])
        for k,v in h.history.iteritems():
            history[k] = history.get(k,[]) + v
        with open('data/%s.history.pkl'%FN,'wb') as fp:
            pickle.dump(history,fp,-1)
        model.save_weights('data/%s.hdf5'%FN, overwrite=True)
        gensamples(X_train,Y_train,idx2word,batch_size=batch_size)

    # predict
    print 'Prediction starts'
    write_predictions()
