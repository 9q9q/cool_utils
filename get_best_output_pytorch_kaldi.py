"""Given a bunch of *.tra output from PyTorch-Kaldi decoding, get the
transcript with lowest error (custom eval script).

Usage:
python get_best_output_pytorch_kaldi.py --tra_dir=/home/galen/pytorch_kaldi/pytorch-kaldi/exp/coursera_LSTM_mfcc_24_epoch/decode_coursera_test_out_dnn2/scoring/ --words=/home/galen/coursera_kaldi/data/lang/words.txt --truth_file=/home/galen/coursera_kaldi/data/train/text --out=int2sym_test.tsv
"""

import argparse
import logging
import os
import pandas as pd
from shutil import copyfile
import subprocess
import sys
import csv

sys.path.insert(0, "/home/galen/utils/create_hyp_file")
sys.path.insert(0, "/home/galen/utils/eval_copy")
from eval_copy import cal_cser
from create_hyp_file import reformat

INT2SYM = "/home/galen/pytorch_kaldi/pytorch-kaldi/kaldi_decoding_scripts/utils/int2sym.pl"
TMP_OUT = "tmp_out"
TRA_SYM = "tra_sym"


def get_cswer(tra_tsv):
    result = pd.read_csv(tra_tsv, sep="\t")
    return result.apply(cal_cser, axis=1).mean()


def convert_to_tsv(tra_file, truth_file, words_file):
    """Convert to sym and save as tsv."""
    subprocess.call("cat {} | perl {} -f 2- {} > {}".format(tra_file, INT2SYM,
        words_file, TRA_SYM), shell=True)
    tsv_out = os.path.join(TMP_OUT, os.path.basename(tra_file))
    reformat(TRA_SYM, truth_file, tsv_out)
    return tsv_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tra_dir", type=str, required=True, help="Path to "
            "dir containing *.tra dirs.")
    parser.add_argument("--words", type=str, required=True, help="Path to "
            "data/lang/words.txt file, e.g. data/train/text.")
    parser.add_argument("--truth_file", type=str, required=True, help="Path to "
            "ground truth transcripts file.")
    parser.add_argument("--out", type=str, required=True, help="Path to "
            "output (transcripts with lowest error rate).")
    args = parser.parse_args()

    tra_paths = []
    for f in os.listdir(args.tra_dir):
        if f.endswith(".tra"):
            tra_paths.append(os.path.join(args.tra_dir, f))

    best_error = float("inf")
    best_tra = ""
    for tra in tra_paths:
        tsv = convert_to_tsv(tra, args.truth_file, args.words)
        error = get_cswer(tsv)
        print("error {}".format(error))
        if error < best_error:
            best_error = error
            best_tra = tsv

    copyfile(best_tra, args.out)

    print("Best hyp with code-switch WER {} saved to {}.".format(best_error,
        args.out))


if __name__ == "__main__":
    main()
