""""Given a course from Coursera test set, extract an adaptation set.

Usage:
    python create_adapt_set_text.py --text=/home/galen/new_cool_adapt_kaldi/data/train/text --wavs=/home/galen/new_cool_adapt_kaldi/data/train/wav.scp --text_out=adapt_text --class_prefix=CSIE1310 --audio_prefixes=0000
"""
import argparse
from sklearn.model_selection import train_test_split
import os


def get_utt_ids(wavs, class_prefix, audio_prefixes):
    ids = []
    class_prefixes = [os.path.join(class_prefix,
        "audio-{}".format(audio_prefix)) for audio_prefix in audio_prefixes]
    with open(wavs) as f:
        for line in f:
            for class_prefix in class_prefixes:
                if class_prefix in line:
                    ids.append(line.split(" ")[0])
    return ids


def create_text_list(utt_ids, text_file):
    feats = []
    with open(text_file) as f:
        for line in f:
            # this could be a dict for speedup but there won't be that much data
            if line.split(" ")[0] in utt_ids:
                feats.append(line)
    return sorted(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Path to "
            "original text for one course.")
    parser.add_argument("--wavs", type=str, required=True, help="Path to "
            "original wav.scp (contains orig filenames).")
    parser.add_argument("--text_out", type=str, required=True, help="Path to "
            "output text.")
    parser.add_argument("--audio_prefixes", type=str, required=True,
            help="audio-${audio_prefix}-seg-... Comma-separated.")
    parser.add_argument("--class_prefix", type=str, required=True,
            help="e.g. test/course-00")
    args = parser.parse_args()

    audio_prefixes = args.audio_prefixes.split(",")
    ids = get_utt_ids(args.wavs, args.class_prefix, audio_prefixes)
    text = create_text_list(ids, args.text)

    with open(args.text_out, "w") as f:
        for t in text:
            f.write(t)


if __name__ == "__main__":
    main()
