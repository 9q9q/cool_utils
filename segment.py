"Copied from jun wei script"

import jieba
import argparse
import os
from multiprocessing import Pool, cpu_count


BASE_DIR = '/home/b06901141/video-subtitle-generator--b06901141/'

# load stopwords set
stopword_set = set()
with open(os.path.join(BASE_DIR, 'material/jieba_dict/stopwords.txt'), 'r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))

punctuation_set = set()
with open(os.path.join(BASE_DIR, 'material/jieba_dict/punctuations.txt'), 'r', encoding='utf-8') as punctuations:
    for punctuation in punctuations:
        punctuation_set.add(punctuation.strip('\n'))

def cut(sentence):
    sentence = sentence.strip('\n').upper()
    output = ""
    words = jieba.cut(sentence, cut_all=False)
    for word in words:
        word = "".join(word.split(" "))
        if word not in punctuation_set:
            output += word + ' '
    output = output[:-1] + '\n'
    return output


def cutHungyi(sentence):
    sentence = sentence.replace('\n', "")
    output = ""
    words = jieba.cut(sentence, cut_all=False)
    for word in words:
        if word not in punctuation_set:
            output += word + ' '
    output = " ".join(output.rstrip(" ").split()) + '\n'
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--hungyi', action='store_false')
    args = parser.parse_args()
    # jieba custom setting.
    jieba.set_dictionary(os.path.join(BASE_DIR,
        'material/jieba_dict/dict.txt.big'))
    print("Start cuting text from %s..." % args.input)
    with open(args.input, 'r', encoding='utf-8') as content:
        output_data = []
        p = Pool(processes=args.njobs)
        if args.hungyi:
            output_data = p.map(cutHungyi, content)
        else:
            output_data = p.map(cut, content)
        p.close()
        p.join()
    with open(args.output, 'w', encoding='utf-8') as fout:
        for data in output_data:
            fout.write(data)
    print("Done")


if __name__ == '__main__':
    main()
