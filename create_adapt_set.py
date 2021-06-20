""""Given a course from Coursera test set, extract an adaptation set.

Usage:
    python create_adapt_set.py --feats=/home/galen/coursera_kaldi/fmllr/course_00.scp --wavs=/home/galen/coursera_kaldi/fmllr/wav.scp --test_out=adapt_set_test1 --train_out=adapt_set_train1 --dev_out=adapt_set_dev1 --class_prefix=test/course00 --audio_prefixes=0000
"""
import argparse
from sklearn.model_selection import train_test_split
import os


def get_utt_ids(wavs, class_prefix, audio_prefixes):
    train_ids = set()
    test_ids = set()
    class_prefixes = [os.path.join(class_prefix,
        "audio-{}".format(audio_prefix)) for audio_prefix in audio_prefixes]
    with open(wavs) as f:
        for line in f:
            for class_prefix in class_prefixes:
                if class_prefix in line:
                    train_ids.add(line.split(" ")[0])
                else:
                    test_ids.add(line.split(" ")[0])
    test_ids = test_ids - train_ids # possible repeats
    return list(train_ids), list(test_ids)


def create_new_feats_list(utt_ids, feats_file):
    feats = []
    with open(feats_file) as f:
        for line in f:
            # this could be a dict for speedup but there won't be that much data
            if line.split(" ")[0] in utt_ids:
                feats.append(line)
    return sorted(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", type=str, required=True, help="Path to "
            "original feats.scp for one course.")
    parser.add_argument("--wavs", type=str, required=True, help="Path to "
            "original wav.scp (contains orig filenames).")
    parser.add_argument("--train_out", type=str, required=True, help="Path to "
            "output train feats.scp.")
    parser.add_argument("--test_out", type=str, required=True, help="Path to "
            "output test feats.scp.")
    parser.add_argument("--dev_out", type=str, required=True, help="Path to "
            "output dev feats.scp.")
    parser.add_argument("--audio_prefixes", type=str, required=True,
            help="audio-${audio_prefix}-seg-... Comma-separated.")
    parser.add_argument("--class_prefix", type=str, required=True,
            help="e.g. test/course-00")
    args = parser.parse_args()

    audio_prefixes = args.audio_prefixes.split(",")

    train_ids, test_ids = get_utt_ids(args.wavs, args.class_prefix,
            audio_prefixes)
    train_ids, dev_ids = train_test_split(train_ids, test_size=0.1)
    
    train_feats_list = create_new_feats_list(train_ids, args.feats)
    test_feats_list = create_new_feats_list(test_ids, args.feats)
    dev_feats_list = create_new_feats_list(dev_ids, args.feats)

    with open(args.train_out, "w") as f:
        for feat in train_feats_list:
            f.write(feat)
    with open(args.test_out, "w") as f:
        for feat in test_feats_list:
            f.write(feat)
    with open(args.dev_out, "w") as f:
        for feat in dev_feats_list:
            f.write(feat)


if __name__ == "__main__":
    main()
