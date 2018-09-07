'''
Sample generator

Sumana Basu
260727568
'''

from utils import log_slack,lpadd,keras_rnn_predict,vocab_fold,vocab_unfold
import sys
import Levenshtein
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils

# generate samples for training
def gensamples(X_test,Y_test,idx2word,skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    i = random.randint(0,len(X_test)-1)
    log_slack('HEAD:{}'.format(' '.join(idx2word[w] for w in Y_test[i][:maxlenh])))
    log_slack('DESC:{}'.format(' '.join(idx2word[w] for w in X_test[i][:maxlend])))
    sys.stdout.flush()

    log_slack('HEADS:')
    x = X_test[i]
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                log_slack('{}, {}'.format(score, ' '.join(words)))
        #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
        else:
                log_slack('{}, {}'.format(score, ' '.join(words)))
        codes.append(code)

# generate samples for testing
def predsamples(X_test,Y_test,idx2word,skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True,writer=None):
    all_heads = {}
    all_preds = {}
    back_skips = skips
    for i in range(819,len(X_test)):
        log_slack('Writing record {}'.format(i))
        #i = random.randint(0,len(X_test)-1)
        all_heads[i] = ' '.join(idx2word[w] for w in Y_test[i][:maxlenh])
        #print 'DESC:',' '.join(idx2word[w] for w in X_test[i][:maxlend])
        #sys.stdout.flush()
        all_preds[i] = []
        #print 'HEADS:'
        x = X_test[i]
        samples = []
        if maxlend == 0:
            skips = [0]
        else:
            skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
        for s in skips:
            start = lpadd(x[:s])
            fold_start = vocab_fold(start)
            #print 'HEAD:',' '.join(idx2word[w] for w in fold_start)
            sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, k=k, temperature=temperature, use_unk=use_unk)
            #print 'beam'
            assert all(s[maxlend] == eos for s in sample)
            samples += [(s,start,scr) for s,scr in zip(sample,score)]

        samples.sort(key=lambda x: x[-1])
        codes = []
        for sample, start, score in samples:
            code = ''
            words = []
            sample = vocab_unfold(start, sample)[len(start):]
            for w in sample:
                if w == eos:
                    break
                words.append(idx2word[w])
                code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
            if short:
                distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
                if distance > -0.6:
                    all_preds[i].append(' '.join(words))
            #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
            else:
                    all_preds[i].append(' '.join(words))
            codes.append(code)
        writer.writerow([all_heads[i],';'.join(all_preds[i])])
        skips = back_skips

def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty]*maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)

    return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxint)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)
