'''
utility functions

Sumana Basu
260727568
'''

# slack env variables
from secrets import slack
from slacker import Slacker
# Ignore if not using slack
slackClient = Slacker(slack)

def log_slack(msg):
    print(msg)
    try:
        slackClient.chat.post_message('#ml',msg,as_user='mimi')
    except Exception as e:
        #print('Slack connectivity issue')
        pass

def str_shape(x):
    return 'x'.join(map(str,x.shape))

# inspect the lstm model layer by layer
def inspect_model(model):
    for i,l in enumerate(model.layers):
        print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
        weights = l.get_weights()
        for weight in weights:
            print str_shape(weight),
        print

# pad the sentence with empty and eos characters
def lpadd(x, maxlend=maxlend, eos=eos):
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

# run the knn prediction
def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    sample_lengths = map(len, samples)
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([prob[sample_length-maxlend-1] for prob, sample_length in zip(probs, sample_lengths)])

# folding operation
def vocab_fold(xs):
    xs = [x if x < oov0 else glove_idx2idx.get(x,x) for x in xs]
    outside = sorted([x for x in xs if x >= oov0])
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

# unfolding operation
def vocab_unfold(desc,xs):
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]
