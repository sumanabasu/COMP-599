'''
This code calculates BLEU score over test set

Sumana Basu
260727568
'''
import pandas as pd
import nltk

# generate bleu score
def bleu():
    bl = []
    df = pd.read_csv('results_2.csv')
    origs = df.original.values
    references = [t.split() for t in origs]
    preds = df.predicted.values
    preds_all = [str(t).split(';') for t in preds]
    for i,r in enumerate(preds_all):
        bs = 0
        ct = 0
        print i
        for t in r:
            hyp = t.split()
            if len(hyp) > 0:
                #print hyp
                try:
                    bs += nltk.translate.bleu_score.sentence_bleu(references, hyp, weights = (0.5, 0.5))
                    ct += 1
                except:
                    bs += 0
        if ct!=0:
            bs = bs / ct
        bl.append(bs)
    return bl
