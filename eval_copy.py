"""
Copy of /home//video-subtitle-generator/src/end2end/eval.py.
Prints error rates.

Usage: python eval_copy.py --file=res.tsv
"""

import argparse
import numpy as np
import pandas as pd
import editdistance as ed

def isChinese(word):
    for char in word:
        if char < u'\u4e00' or char > u'\u9fa5':
            return False
    return True

SEP = ' '

# Arguments
# parser = argparse.ArgumentParser(
    # description='Script for evaluating recognition results.')
# parser.add_argument('--file', type=str, help='Path to result csv.')
# paras = parser.parse_args()

# Error rate functions


def cal_cer(row):
    return 100*float(ed.eval(str(row.hyp), str(row.truth)))/len(str(row.truth))

def cal_ch_cer(row):
    hyp = str(row.hyp)
    truth = str(row.truth)
    ch_hyp = ""
    ch_truth = ""
    for c in hyp:
        if isChinese(c):
            ch_hyp += c
    for c in truth:
        if isChinese(c):
            ch_truth += c
    length = len(ch_truth) if len(ch_truth) > 0 else 1
    return 100*float(ed.eval(ch_hyp, ch_truth)/length)

def cal_cser(row):
    hyp_sent = str(row.hyp).split(SEP)
    truth_sent = str(row.truth).split(SEP)
    hyp_list = []
    truth_list = []
    for w in hyp_sent:
        if isChinese(w):
            hyp_list.extend(list(w))
        else:
            hyp_list.append(w)
    for w in truth_sent:
        if isChinese(w):
            truth_list.extend(list(w))
        else:
            truth_list.append(w)
    length = len(truth_list) if len(truth_list) > 0 else 1
    return 100*float(ed.eval(hyp_list, truth_list))/length

def cal_wer(row):
    return 100*float(ed.eval(str(row.hyp).split(SEP), str(row.truth).split(SEP)))/len(str(row.truth).split(SEP))


# # Evaluation
# result = pd.read_csv(paras.file, sep='\t')
# result['hyp_char_cnt'] = result.apply(lambda x: len(str(x.hyp)), axis=1)
# result['hyp_word_cnt'] = result.apply(lambda x: len(str(x.hyp).split(SEP)), axis=1)
# result['truth_char_cnt'] = result.apply(lambda x: len(str(x.truth)), axis=1)
# result['truth_word_cnt'] = result.apply(
    # lambda x: len(str(x.truth).split(SEP)), axis=1)
# result['cer'] = result.apply(cal_cer, axis=1)
# result['wer'] = result.apply(cal_wer, axis=1)
# result['ch_cer'] = result.apply(cal_ch_cer, axis=1)
# result['cser'] = result.apply(cal_cser, axis=1)

# # Show results
# print()
# print('============  Result of', paras.file, '============')
# print(' -----------------------------------------------------------------------')
# print('| Statics\t\t|  Truth\t|  Prediction\t| Abs. Diff.\t|')
# print(' -----------------------------------------------------------------------')
# print('| Avg. # of chars\t|  {:.2f}\t|  {:.2f}\t|  {:.2f}\t\t|'.
      # format(result.truth_char_cnt.mean(), result.hyp_char_cnt.mean(),
             # np.mean(np.abs(result.truth_char_cnt-result.hyp_char_cnt))))
# print('| Avg. # of words\t|  {:.2f}\t|  {:.2f}\t|  {:.2f}\t\t|'.
      # format(result.truth_word_cnt.mean(), result.hyp_word_cnt.mean(),
             # np.mean(np.abs(result.truth_word_cnt-result.hyp_word_cnt))))
# print(' -----------------------------------------------------------------------')
# print(' ---------------------------------------------------------------')
# print('| Error Rate (%)| Mean\t\t| Std.\t\t| Min./Max.\t|')
# print(' ---------------------------------------------------------------')
# print('| Character\t\t| {:2.4f}\t| {:.2f}\t\t| {:.2f}/{:.2f}\t|'.format(result.cer.mean(), result.cer.std(),
                                                                      # result.cer.min(), result.cer.max()))
# print('| Word\t\t\t| {:2.4f}\t| {:.2f}\t\t| {:.2f}/{:.2f}\t|'.format(result.wer.mean(), result.wer.std(),
                                                                   # result.wer.min(), result.wer.max()))
# print('| Chinese Character\t| {:2.4f}\t| {:.2f}\t\t| {:.2f}/{:.2f}\t|'.format(result.ch_cer.mean(), result.ch_cer.std(),
                                                                      # result.ch_cer.min(), result.ch_cer.max()))
# print('| Code Swith Error Rate\t| {:2.4f}\t| {:.2f}\t\t| {:.2f}/{:.2f}\t|'.format(result.cser.mean(), result.cser.std(),
                                                                      # result.cser.min(), result.cser.max()))
# print(' ---------------------------------------------------------------')
# print('Note : If the text unit is phoneme, WER = PER and CER is meaningless.')
# print()
