"""Converts Chinese to traditional, capitalize English, numbers to Chinese 
characters."""

import argparse
import logging
import os
import re
import sys
import csv
# location of num2chinese
sys.path.insert(1, "/home/b06901141/video-subtitle-generator--b06901141/src/")
from num2chinese import num2chinese
import opencc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input text file "
            "(Kaldi standard text format).")
    parser.add_argument("--output", type=str, required=True, help="Output "
        "normalized text file.")
    args = parser.parse_args()

    with open(args.input) as f:
        text = list(f)

    converter = opencc.OpenCC("s2t.json")
    with open(args.output, "w") as f:
        for utt in text:
            utt = utt.strip()
            print(utt)
            if utt:
                utt_id, text = utt.split(" ", 1)

            # convert only num substrings
            nums = []
            for match in re.finditer(r'[0-9]+', text):
                print(match)
                nums.append(match.span())
            for num in nums:
                text[num[0]:num[1]] = num2chinese(
                        text[num[0]:num[1]], simp=False) 

            normalized_text = converter.convert(text.upper())
            f.write(utt_id + " " + normalized_text + "\n")


if __name__ == "__main__":
    main()
