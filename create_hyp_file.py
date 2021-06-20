"""Given labels and hyps with corresponding utts, create tsv output that can be
evaluated with
/home/b06901141/video-subtitle-generator--b06901075/src/end2end/eval.py.

Usage:
python create_hyp_file.py --hyp_file=/home/galen/pytorch_kaldi/pytorch-kaldi/kaldi_decoding_scripts/test_4_tra --truth_file=/home/galen/coursera_kaldi/data/train/text --out=out.tsv
"""

import argparse
import copy
import logging
import os
import sys
import csv


def reformat(hyp_file, truth_file, out):
    utts = {}

    with open(hyp_file) as f:
        for line in f:
            utt_id, text = line.split(" ", 1)
            utts[utt_id] = [text.strip()]
    
    with open(truth_file) as f:
        for line in f:
            utt_id, text = line.split(" ", 1)
            if utt_id in utts:
                utts[utt_id].append(text.strip())

    utts_copy = copy.deepcopy(utts)
    for utt_id, text in utts_copy.items():
        if len(text) != 2:
            utts.pop(utt_id)

    with open(out, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["idx", "hyp", "truth"])
        for utt_id, text in utts.items():
            writer.writerow([utt_id, text[0], text[1]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_file", type=str, required=True, help="Path to "
            "labels. Format: Each line has utt id, space, then the text.")
    parser.add_argument("--hyp_file", type=str, required=True, help="Path to "
            "hyps. Format: Each line has utt id, space, then the text.")
    parser.add_argument("--out", type=str, required=True, help="Path to "
            "output. Format: Each line has utt id, hyp, truth separated by "
            "tabs.")
    args = parser.parse_args()

    reformat(args.hyp_file, args.truth_file, args.out)


if __name__ == "__main__":
    main()
