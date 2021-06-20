"""Clean up SRT files that have formatting issues."""

import os
from pathlib import Path
import re
import webvtt

BASEDIR="/home/galen/data/5.4_new_data/WRITE5019-01/transcript"


# also checks timestamps
def fix_ids(whole_file):
    digit_lines = []
    for i, line in enumerate(whole_file):
        if re.match(r"^\d+ *$", line):
            digit_lines.append(i)

    # replace digits with sequential ones
    num_entries = len(digit_lines)
    ordered_digits = range(1, num_entries+1)
    for i in range(num_entries):
        whole_file[int(digit_lines[i])] = str(ordered_digits[i]) + "\n"
    return whole_file


def get_segments(srt):
    transcript_lines = []
    complete_segments = []
    with open(srt) as f:
        whole_file = f.readlines()
    for i, line in enumerate(whole_file):
        if line and re.match(r"^.*[^0-9\n\s]+.*$", line) and "-->" not in line:
            transcript_lines.append(i)
    for transcript_line in transcript_lines:
        if "-->" in whole_file[transcript_line-1] and re.match(r"^\d+ *$",
                whole_file[transcript_line-2]):
            complete_segments.extend(whole_file[transcript_line-2:transcript_line+1])
            complete_segments[-1] += "\n"
    return complete_segments


def main():
    srts = list(Path(BASEDIR).rglob("*.srt"))
    for srt in srts:
        fixed_segments = get_segments(srt)
        fixed = fix_ids(fixed_segments)
        with open(srt, "w") as f:
            whole_file_str = "".join(fixed).lstrip() # lstrip to remove leading
            f.write(whole_file_str)


if __name__ == '__main__':
    main()
