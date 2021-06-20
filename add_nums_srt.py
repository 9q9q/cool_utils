"""Add IDs to SRT files with missing IDs."""

import os
from pathlib import Path
import re
import webvtt

# SRT="/home/galen/data/5.4_new_data/WRITE5019-01/transcript/Autobiography_2.srt"
SRT="/home/galen/data/5.4_new_data/WRITE5019-01/transcript/Autobiography_4.srt"


# also checks timestamps
def add_ids(whole_file):
    space_lines = []
    for i, line in enumerate(whole_file):
        if re.match(r"\s", line):
            space_lines.append(i)

    # replace digits with sequential ones
    num_entries = len(space_lines)
    ordered_digits = range(1, num_entries+1)
    for i in range(num_entries):
        whole_file[int(space_lines[i])] = "\n" + str(ordered_digits[i]) + "\n"
    return whole_file

def main():
    # srts = list(Path(BASEDIR).rglob("*.srt"))
    with open(SRT) as f:
        whole_file = f.readlines()
    fixed = add_ids(whole_file)
    with open(SRT, "w") as f:
        whole_file_str = "".join(fixed).lstrip() # lstrip to remove leading
        f.write(whole_file_str)


if __name__ == '__main__':
    main()
