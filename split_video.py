import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import Pool
import webvtt
import os
import librosa
import datetime
import numpy as np
import soundfile as sf
import json
from collections import defaultdict
import jieba
import re
import argparse
from opencc import OpenCC
import pandas as pd
from tqdm import tqdm
import sys


# BASE_DIR="/home/b06901141/video-subtitle-generator--b06901141/material"
BASE_DIR="/home/galen/video-subtitle-generator/material"


stopword_set = set()
with open(os.path.join(BASE_DIR, "jieba_dict/stopwords.txt"), "r", encoding="utf-8") as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip("\n"))

punctuation_set = set(" ")
with open(os.path.join(BASE_DIR, "jieba_dict/punctuations.txt"), "r", encoding="utf-8") as punctuations:
    for punctuation in punctuations:
        punctuation_set.add(punctuation.strip("\n"))

englist_punctuations = punctuation_set.copy()
englist_punctuations.discard("'")
englist_punctuations.discard(" ")
englist_punctuations.discard("<")
englist_punctuations.discard(">")

t_gnd = datetime.datetime(1900, 1, 1, 0, 0)  # for reset time
np.random.seed(0)
dataset_root = "coursera"
destination = "splitted-coursera"
a = "\[(.+?)\]"
regex = re.compile(a)
# discourse = "(\[|\??(.+?)(\]|\??"
# discourse_regex = re.compile(discourse)
hesitation = "\((.+?)\)"
hesitation_regex = re.compile(hesitation)
other_lang = "\#(.+?)\#"
other_lang_regex = re.compile(other_lang)
opencc = OpenCC("s2tw")
brackets = "\<(.+?)\>"
brackets_regex = re.compile(brackets)
noise = "\{(.+?)\}"
noise_regex = re.compile(noise)
# train set, contrain all multi speaker course
# dev set seen speaker 1 female speaker, 2 male speaker
# test set unseen speaker 1 female speaker, 2 male speaker
def split_course():
    with open("coursera-info.json", "r") as f:
        data = json.load(f)
    english_courses = ["3d-cad-application", "3d-cad-fundamental", "bim-application", "bim-fundamentals"]
    multi_instructor = []
    female_single_course = []
    female_multi_course = []
    male_single_course = []
    male_multi_course = []
    instructor_course = defaultdict(list)
    instructor_gender = {}
    for course, info in data.items():
        if course in english_courses:
            continue
        if len(info["instructor"]) > 1:
            multi_instructor.append(course)
        else:
            instructor = info["instructor"][0]
            gender = info["gender"]
            instructor_course[instructor].append(course)
            instructor_gender[instructor] = gender
    
    for instructor, courses in instructor_course.items():
        if len(courses) > 1:
            if instructor_gender[instructor] == "male":
                male_multi_course.extend(courses)
            elif instructor_gender[instructor] == "female":
                female_multi_course.extend(courses)
        else:
            if instructor_gender[instructor] == "male":
                male_single_course.extend(courses)
            elif instructor_gender[instructor] == "female":
                female_single_course.extend(courses)
    # print(female_single_course)
    # print(female_multi_course)
    # print(male_single_course)
    # print(male_multi_course)
    train_courses = []
    np.random.shuffle(female_single_course)
    np.random.shuffle(female_multi_course)
    np.random.shuffle(male_single_course)
    np.random.shuffle(male_multi_course)
    np.random.shuffle(english_courses)
    dev_courses = [female_multi_course[-1], male_multi_course[0], male_multi_course[1]]
    test_courses = [female_single_course[-1], male_single_course[0], male_single_course[1]]
    train_courses.extend(multi_instructor)
    train_courses.extend(female_multi_course[:-1])
    train_courses.extend(female_single_course[:-1])
    train_courses.extend(male_multi_course[2:])
    train_courses.extend(english_courses)
    train_courses.extend(male_single_course[2:])
    np.random.shuffle(train_courses)
    return train_courses, dev_courses, test_courses

def extract(filename):
    pass
    #mp4 = str(filename)
    #wav = mp4.replace(".mp4", ".wav")
    #subprocess.call(("ffmpeg", "-y", "-i", mp4, "-ar", "32000", "-ac", "2", "-f", "wav", wav))
    #subprocess.call(("rm", mp4))

def translate(words):
    tw_words = []
    for word in words:
        tw_word = opencc.convert(word)
        if len(tw_word) < len(word):
            tw_c = ""
            for c in word:
                tw_c += opencc.convert(c)
            tw_words.append(tw_c)
        else:
            tw_words.append(tw_word)
    txt = " ".join(tw_words)
    return txt


def cut(sentence):
    sentence = sentence.replace("\n", "")
    sentence = regex.sub("", sentence)
    output = ""
    words = jieba.cut(sentence, cut_all=False)
    for word in words:
        if word not in punctuation_set:
            output += word.upper() + " "
    output = output.rstrip(" ").split()
    output = translate(output)
    return output

def read_libri_text(file):
    file = str(file)
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]

def coursera_splitText(subtitleName, output_dir, fid):
    # subtitleName = str(subtitleName)
    wavName = subtitleName.replace(".zh-TW.srt", "")
    wavName = wavName.replace(".en.srt", "")
    wavName = wavName.replace(".zh-TW.vtt", "")
    wavName = wavName.replace(".srt", "")
    # print(wavName)

    # Parse subtitle file
    try:
        if subtitleName.endswith("srt"):
            subtitle = webvtt.from_srt(subtitleName)
        elif subtitleName.endswith("vtt"):
            subtitle = webvtt.read(subtitleName)
    except:
        return "", 0
    if not os.path.exists(wavName + ".wav"):
        return "", 0
    else:
        wav, sr = librosa.load(os.path.join(
            wavName) + ".wav", sr=None)
    #subprocess.call(("rm", wavName + ".wav"))
    text = ""
    total_time = 0.0
    wavName = "_".join(wavName.split(" "))
    sub_filename = os.path.split(wavName)[-1]
    for index, i in enumerate(subtitle):
        parsedText = cut(i.text)
        if parsedText == "":
            print("No word skipped")
            continue
        start = (datetime.datetime.strptime(i.start, "%H:%M:%S.%f") -
                 t_gnd).total_seconds()  # sentence start time in seconds
        end = (datetime.datetime.strptime(i.end, "%H:%M:%S.%f") -
               t_gnd).total_seconds()  # sentence end time in seconds
        if end - start > 20:
            print("Skip Large audio file")
            continue
        if end - start < 0.2:
            print("Skip small audio file")
            continue
        output_txt = os.path.join(output_dir,
                "audio-{:04}-seg-{:05}.txt".format(fid, index))
        output_wav = os.path.join(output_dir,
                "audio-{:04}-seg-{:05}.wav".format(fid, index))
        
        with open(output_txt, "w") as fout:
            fout.write("{}\n".format(parsedText))
        # fileName = os.path.join(output_dir, "%s-%d.wav" % (sub_filename, index))
        # change start, end into int index
        
        
        total_time += (end - start)

        start, end = (np.array([start, end]) * sr).astype(np.int32)
        segment = wav[start:end+1]
        sf.write(output_wav, segment, sr, format="wav")
        text += "{}\n".format(parsedText)
    return text, total_time

def process_seame_text(txt):
    txt = discourse_regex.sub("", txt)
    txt = hesitation_regex.sub("", txt)
    txt = other_lang_regex.sub("", txt)
    txt = txt.strip(" ")
    words = txt.split()
    tw_words = []
    for word in words:
        tw_word = opencc.convert(word)
        if len(tw_word) < len(word):
            tw_c = ""
            for c in word:
                tw_c += opencc.convert(c)
            tw_words.append(tw_c)
        else:
            tw_words.append(tw_word)
    output = ""
    for c in " ".join(tw_words):
        if c not in englist_punctuations:
            output += c.upper()
    output = " ".join(output.rstrip(" ").split())
    output = output.replace("<UNK>", "<unk>")
    return output

def new_cool_split_text(subtitleName, output_dir, fid):
    wavName = subtitleName.replace(".mp4.srt", "")
    # wavName = subtitleName.replace(".srt", "") # this was needed for some of the videos bc of different naming
    wavName = wavName.replace("/transcript/", "/video/")
    print("wavname: {}".format(wavName))

    # only sometimes necessary
    # videoName = subtitleName.replace(".srt", ".wav").replace("/transcript/", "/video/")
    # if not (os.path.isfile(subtitleName) and os.path.isfile(videoName)):
        # exit("{} and {} don't both exist".format(subtitleName, videoName))

    # Parse subtitle file
    try:
        subtitle = webvtt.from_srt(subtitleName)
    except Exception as e:
        exit("webvtt error: {}".format(e))
        # return "", 0
    if not os.path.exists(wavName + ".wav"):
        return "", 0
    else:
        wav, sr = librosa.load(os.path.join(
            wavName) + ".wav", sr=None)
    #subprocess.call(("rm", wavName + ".wav"))
    text = ""
    total_time = 0.0
    wavName = "_".join(wavName.split(" "))
    sub_filename = os.path.split(wavName)[-1]
    for index, i in enumerate(subtitle):
        parsedText = cut(i.text)
        if parsedText == "":
            print("No word skipped")
            continue
        start = (datetime.datetime.strptime(i.start, "%H:%M:%S.%f") -
                 t_gnd).total_seconds()  # sentence start time in seconds
        end = (datetime.datetime.strptime(i.end, "%H:%M:%S.%f") -
               t_gnd).total_seconds()  # sentence end time in seconds
        if end - start > 20:
            print("Skip Large audio file")
            continue
        if end - start < 0.2:
            print("Skip small audio file")
            continue
        output_txt = os.path.join(output_dir,
                "audio-{:04}-seg-{:05}.txt".format(fid, index))
        output_wav = os.path.join(output_dir,
                "audio-{:04}-seg-{:05}.wav".format(fid, index))
        
        with open(output_txt, "w") as fout:
            fout.write("{}\n".format(parsedText))
        # fileName = os.path.join(output_dir, "%s-%d.wav" % (sub_filename, index))
        # change start, end into int index
        
        
        total_time += (end - start)

        start, end = (np.array([start, end]) * sr).astype(np.int32)
        segment = wav[start:end+1]
        sf.write(output_wav, segment, sr, format="wav")
        text += "{}\n".format(parsedText)
    return text, total_time


def seame_splitText(flacFile, output_dir, fid):
    paths = flacFile.split('/')
    paths[-2] = "transcript"
    paths.insert(-1, "phaseII")
    txt = "/".join(paths).replace(".flac", ".txt")
    if not os.path.exists(txt):
        return "", 0
    print(flacFile)
    y, sr = librosa.load(flacFile, sr=None)
    transcript = pd.read_csv(txt, sep="\t", header=None)
    start = transcript[1]
    end = transcript[2]
    text_col = transcript[4]
    processed_text = text_col.apply(process_seame_text)
    total_time = 0.0
    total_text = ""
    for i, (s, e, text) in enumerate(zip(start, end, processed_text)):
        if len(text) == 0 or text == "\n":
            continue
        if e - s < 200:
            continue
        if e-s > 30000:
            continue
        start_index = s * sr // 1000
        end_index = e * sr // 1000
        audio_seg = y[start_index:end_index+1]
        total_time += (e - s) / 1000
        output_txt = os.path.join(output_dir, "audio-{:04}-seg-{:05}.txt".format(fid, i))
        output_wav = os.path.join(output_dir, "audio-{:04}-seg-{:05}.wav".format(fid, i))
        sf.write(output_wav, audio_seg, sr, format="wav")
        with open(output_txt, "w") as f:
            total_text += "{}\n".format(text)
            f.write("{}\n".format(text))
    return total_text, total_time



def process_hkust_text(txt):
    split_txt = txt.split(" ")
    start, end = float(split_txt[0]), float(split_txt[1])
    channel = 0 if "A:" == split_txt[2] else 1
    txt = " ".join(split_txt[3:])
    # txt = txt.replace("%??, "").replace("%??, "").replace("%??, "") # remove hesitation
    txt = brackets_regex.sub("", txt) # remove speaker noise
    txt = noise_regex.sub("", txt) # remove speaker noise
    txt = txt.strip(" ")
    words = cut(txt).split(" ")
    txt = translate(words)
    return start, end, channel, txt.upper()


def hkust_splitText(sphFile, output_dir, fid):
    paths = sphFile.split('/')
    paths[0] += "tr"
    paths[-3] = "trans"
    txt = "/".join(paths).replace(".sph", ".txt")
    # print(txt)
    if not os.path.exists(txt):
        return "", 0
    print(sphFile)
    y, sr = sf.read(sphFile)
    # channel_A = y[:, 0]
    # channel_B = y[:, 1]
    with open(txt, "r") as f:
        data = f.read().strip("\n").split("\n\n")[1:]
    processed_text = [process_hkust_text(text) for text in data]
    total_time = 0.0
    total_text = ""
    for i, (s, e, ch, text) in enumerate(processed_text):
        if len(text) == 0 or text == "\n":
            continue
        if (e-s) < 0.2:
            continue
        if (e-s) > 20:
            continue
        start_index = int(s * sr)
        end_index = int(e * sr)
        audio_seg = y[start_index:end_index+1, ch]
        total_time += (e - s)
        output_txt = os.path.join(output_dir, "audio-{:04}-seg-{:05}.txt".format(fid, i))
        output_wav = os.path.join(output_dir, "audio-{:04}-seg-{:05}.wav".format(fid, i))
        sf.write(output_wav, audio_seg, sr, format="wav")
        with open(output_txt, "w") as f:
            total_text += "{}\n".format(text)
            f.write("{}\n".format(text))
    return total_text, total_time

def process_aishell_text(txt):
    txt = cut(txt)
    txt = translate(txt.split(" "))
    return txt

def verbose_time(set_name, total_time):
    hours = int(total_time / 3600)
    time_left = int(total_time) % 3600
    minutes = int(time_left / 60)
    time_left = time_left % 60
    seconds = time_left
    print("{} Total Time: {} hours, {} minutes, {} seconds".format(set_name, hours, minutes, seconds))

def exec_cmd(cmd):
    subprocess.call(cmd)

def perturb_eatmic(input_dir, methods, output_dir):
    commands = []
    wavs = list(Path(input_dir).rglob("*.wav"))
    for method in methods:
        for wav in wavs:
            wav = str(wav)
            txt = wav + ".lab"
            target_wav = os.path.join(output_dir, wav.split("/")[-1])
            target_wav = target_wav.replace(".wav", ".{}.wav".format(method))
            target_txt = target_wav.replace(".wav", ".txt")
            with open(txt, "r") as f:
                data = f.read().strip("\n").split("\n")[-1]
            with open(target_txt, "w") as f:
                f.write(data + "\n")
            # commands.append(["cp", txt, target_txt])
            if method == "pitch":
                new_rate = np.random.randint(-50, 51)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
            elif method == "denoise":
                subprocess.call(["sox", wav, "-n", "trim", "noiseprof", "{}.prof".format(wav)])
                subprocess.call(["sox", wav, target_wav, "noisered", "{}.prof".format(wav), "0.02"])
                subprocess.call(["rm", "{}.prof".format(wav)])
            elif method == "none":
                commands.append(["cp", str(wav), target_wav])
            else:
                new_rate = np.random.uniform(0.9, 1.1)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
    return commands



def perturb_other(input_dir, methods, output_dir, stats):
    commands = []
    wavs = list(Path(input_dir).rglob("*.wav"))
    for wav in wavs:
        wav = str(wav)
        fid = wav.replace(".wav", "")
        chinese_rate = stats[fid][0]
        print(chinese_rate)
        if chinese_rate == 1:
            continue
        if chinese_rate <= 0.8 and chinese_rate > 0.0:
            methods = ["none", "tempo", "speed", "pitch"]
        for method in methods:
            txt = wav.replace(".wav", ".txt")
            target_wav = os.path.join(output_dir, wav.split("/")[-1])
            target_wav = target_wav.replace(".wav", ".{}.wav".format(method))
            target_txt = target_wav.replace(".wav", ".txt")
            with open(txt, "r") as f:
                data = f.read().strip("\n").split("\n")[-1]
            with open(target_txt, "w") as f:
                f.write(data + "\n")
            if method == "pitch":
                new_rate = np.random.randint(-50, 51)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
            elif method == "denoise":
                subprocess.call(["sox", wav, "-n", "trim", "noiseprof", "{}.prof".format(wav)])
                subprocess.call(["sox", wav, target_wav, "noisered", "{}.prof".format(wav), "0.02"])
                subprocess.call(["rm", "{}.prof".format(wav)])
            elif method == "none":
                commands.append(["cp", str(wav), target_wav])
            else:
                new_rate = np.random.uniform(0.9, 1.1)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
    return commands

def perturb_coursera(input_dir, methods, output_dir, stats):
    commands = []
    perturb_method = ["none", "tempo", "speed", "pitch"]
    wavs = list(Path(input_dir).rglob("*.wav"))
    for wav in wavs:
        wav = str(wav)
        fid = wav.replace(".wav", "")
        chinese_rate = stats[fid][0]
        print(chinese_rate)
        if chinese_rate == 0.0:
            continue
        elif chinese_rate == 1:
            r = np.random.random()
            if r > 0.9:
                methods = ["none"]
            else:
                continue
        elif chinese_rate > 0.9:
            methods = set()
            while len(methods) < 2:
                methods.add(np.random.choice(perturb_method))
        else:
            methods = ["none", "tempo", "speed", "pitch"]
        for method in methods:
            txt = wav.replace(".wav", ".txt")
            target_wav = os.path.join(output_dir, wav.split("/")[-1])
            target_wav = target_wav.replace(".wav", ".{}.wav".format(method))
            target_txt = target_wav.replace(".wav", ".txt")
            with open(txt, "r") as f:
                data = f.read().strip("\n").split("\n")[-1]
            with open(target_txt, "w") as f:
                f.write(data + "\n")
            if method == "pitch":
                new_rate = np.random.randint(-50, 51)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
            elif method == "denoise":
                subprocess.call(["sox", wav, "-n", "trim", "noiseprof", "{}.prof".format(wav)])
                subprocess.call(["sox", wav, target_wav, "noisered", "{}.prof".format(wav), "0.02"])
                subprocess.call(["rm", "{}.prof".format(wav)])
            elif method == "none":
                commands.append(["cp", str(wav), target_wav])
            else:
                new_rate = np.random.uniform(0.9, 1.1)
                commands.append(["sox", str(wav), target_wav, method, str(new_rate)])
    return commands

def read_formosa_text(file):
    paths = str(file).split("/")
    paths[-4] = "Text"
    text_file = "/".join(paths).replace(".wav", ".txt")
    with open(text_file, 'r', encoding="utf-8", errors="ignore") as fp:
        return fp.read().strip("\n").upper()

def read_librispeech_text(file):
    src_file = '-'.join(str(file).split('-')[:-1])+'.trans.txt'
    idx = str(file).split('/')[-1].split('.')[0]
    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]

def read_text(file):
    file = str(file)
    if "eatmic" in file.lower():
        return read_eatmic_text(file)
    elif "coursera" in file.lower():
        return read_coursera_text(file)
    elif "ner-trs-all" in file.lower():
        return read_formosa_text(file)
    elif "timit" in file.lower():
        return read_timit_text(file)
    elif "librispeech" in file.lower():
        return read_librispeech_text(file)
    else:
        return read_other_text(file)

def read_librispeech_text(file):
    
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    print(file)
    print(src_file)
    idx = file.split('/')[-1].split('.')[0]
    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]


def read_timit_text(file):
    text_file = file.replace(".wav", ".txt")
    if os.path.exists(text_file):
        with open(text_file, "r") as fp:
            return fp.read().strip("\n").split(" ")[-1].upper()
    else:
        wrd_file = file.replace(".wav", ".wrd")
        with open(wrd_file, "r") as fp:
            data = fp.read().strip("\n").split("\n")
        data = [line.split(" ")[-1].upper() for line in data]
        return " ".join(data)

def read_coursera_text(file):
    split_path = os.path.split(file)
    text_file = file.replace(".wav", ".txt")
    with open(text_file, "r") as fp:
        return fp.read().strip("\n")   

def read_eatmic_text(file):
    text_file = file + ".lab"
    with open(text_file, 'r', encoding="utf-8", errors="ignore") as fp:
        return fp.read().strip("\n").split("\n")[-1]

def read_formosa_text(file):
    paths = file.split("/")
    paths[-4] = "Text"
    text_file = "/".join(paths).replace(".wav", ".txt")
    with open(text_file, 'r', encoding="utf-8", errors="ignore") as fp:
        return fp.read().strip("\n").upper()

def read_other_text(file):
    text_file = file.replace(".wav", ".txt")
    with open(text_file, "r") as fp:
        return fp.read().strip("\n")

def collect_whole_text(folder):
    output_file = os.path.join(folder, "whole_text.txt")
    txts = list(Path(folder).rglob("*.flac"))
    with Pool(16) as pool:
        output_text = pool.map(read_text, txts)
    output_text = list(set(output_text))
    with open(output_file, "w") as f:
        for text in output_text:
            f.write("{}\n".format(text))

def isChinese(word):
    for char in word:
        if char < u'\u4e00' or char > u'\u9fa5':
            return False
    return True


def collect_whole_chinese_text(folder):
    output_file = os.path.join(folder, "whole_text_ch.txt")
    f_cs = open(os.path.join(folder, "whole_text_cs.txt"), "w")
    wavs = list(Path(folder).rglob("*.wav"))
    output_text = []
    for f in tqdm(wavs, ncols=120):
        if "clean" in str(f):
            continue
        if "cool-small" in str(f):
            continue
        if "dev/" in str(f):
            continue
        if "test/" in str(f):
            continue
        output_text.append(read_text(str(f)))
    output_text = list(set(output_text))
    with open(output_file, "w") as f:
        for text in tqdm(output_text, ncols=120):
            ch_cnt = 0
            en_cnt = 0
            for word in text.split(" "):
                if isChinese(word):
                    ch_cnt += len(word)
                else:
                    en_cnt += 1
            cn_rate =  ch_cnt / (ch_cnt + en_cnt + 1e-10)
            if isChinese("".join(text.split(" "))):
                f.write("{}\n".format(text))
            elif cn_rate > 0:
                f_cs.write("{}\n".format(text))


def collect_whole_englist_text(folder):
    output_file = os.path.join(folder, "whole_text_eng.txt")
    # f_cs = open(os.path.join(folder, "whole_text_cs.txt"), "w")
    wavs = list(Path(folder).rglob("*.flac"))
    output_text = []
    for f in tqdm(wavs, ncols=120):
        if "clean" in str(f):
            continue
        if "cool-small" in str(f):
            continue
        if "dev/" in str(f):
            continue
        if "test/" in str(f):
            continue
        output_text.append(read_text(str(f)))
    output_text = list(set(output_text))
    with open(output_file, "w") as f:
        for text in tqdm(output_text, ncols=120):
            f.write("{}\n".format(text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get-raw-text", action="store_true")
    parser.add_argument("--dataset", type=str, default="coursera")
    args = parser.parse_args()
    if args.dataset == "coursera":
        dataset_root = "dataset/coursera"
        destination = "dataset/splitted-coursera"
        with open("coursera-info.json", "r") as f:
            coursera_info = json.load(f)
        with open("dataset/coursera/spk2id.json", "r") as f:
            spk2id = json.load(f)
        
        jieba.set_dictionary(os.path.join(BASE_DIR,"jieba_dict/dict.txt.big"))
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, "train"), exist_ok=True)
        # os.makedirs(os.path.join(destination, "train-small"), exist_ok=True)
        os.makedirs(os.path.join(destination, "dev"), exist_ok=True)
        os.makedirs(os.path.join(destination, "test"), exist_ok=True)
        train_courses, dev_courses, test_courses = split_course()
        print(train_courses, dev_courses, test_courses)
        set_times = []
        train_set_time = 0.0
        for courses, set_name in zip([train_courses, dev_courses, test_courses], ["train", "dev", "test"]):
            audio_id = 0
            course_id = 0
            # if set_name == "train":
            #     output_train_text = open(os.path.join(destination, "train-small", "whole_text.txt"), "w")
            output_whole_text = open(os.path.join(destination, set_name, "whole_text.txt"), "w")
            set_time = 0.0
            for i, course in enumerate(courses):
                info = coursera_info[course]
                gender = info["gender"]
                instructor_id = "_".join(str(spk2id[spk]) for spk in info["instructor"])
                audio_dir = os.path.join(dataset_root, gender, instructor_id, course)
                srt_files = list(Path(audio_dir).rglob("*.zh-TW.srt"))
                if len(srt_files) == 0:
                    srt_files = list(Path(audio_dir).rglob("*.en.srt"))
                text = ""
                srt_files = [str(srt) for srt in srt_files]
                audio_ids = list(range(audio_id, audio_id + 1 + len(srt_files)))
                output_dir = os.path.join(destination, set_name, "course-{:02}".format(i))
                os.makedirs(output_dir, exist_ok=True)
                with Pool(16) as pool:
                    outputs = pool.starmap(coursera_splitText, zip(srt_files, [output_dir]*len(srt_files), audio_ids))
                for filetext, filetime in outputs:
                    text += filetext
                    set_time += filetime
                output_whole_text.write(text)
                # if set_name == "train" and i < 10:
                #     output_train_text.write(text)
                #     subprocess.call(("cp", "-r", output_dir, os.path.join(destination, "train-small")))
                audio_id += len(srt_files)
            set_times.append(set_time)
        set_times.append(train_set_time)
        for time, set_name in zip(set_times, ["train", "dev", "test"]):
            verbose_time(set_name, time)
    elif args.dataset.lower() == "seame":
        # train/dev split is https://github.com/zengzp0912/SEAME-dev-set
        dataset_root = "dataset/seame"
        destination = "dataset/splitted-seame"
        with open("dataset/SEAME-dev-set/dev_man/wav.scp") as dev_man:
            file_list = dev_man.read().strip("\n").split("\n")
            dev_man_id = [line.split(" ")[0] for line in file_list]
        with open("dataset/SEAME-dev-set/dev_sge/wav.scp") as dev_sge:
            file_list = dev_sge.read().strip("\n").split("\n")
            dev_sge_id = [line.split(" ")[0] for line in file_list]
        flac_files = sorted(list(Path(dataset_root).rglob("*.flac")))
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, "train"), exist_ok=True)
        os.makedirs(os.path.join(destination, "dev-man"), exist_ok=True)
        os.makedirs(os.path.join(destination, "dev-sge"), exist_ok=True)
        output_train_file = os.path.join(os.path.join(destination, "train"), "whole_text.txt")
        output_man_file = os.path.join(os.path.join(destination, "dev-man"), "whole_text.txt")
        output_sge_file = os.path.join(os.path.join(destination, "dev-sge"), "whole_text.txt")
        f_train = open(output_train_file, "w")
        f_man = open(output_man_file, "w")
        f_sge = open(output_sge_file, "w")
        total_time = 0.0
        for i, flacFile in enumerate(flac_files):
            fid = str(flacFile).split("/")[-1].replace(".flac", "").lower()
            if fid in dev_sge_id:
                output_dir = os.path.join(os.path.join(destination, "dev-sge"), "audio-{:04}".format(i))
                os.makedirs(output_dir, exist_ok=True)
                text, time = seame_splitText(str(flacFile), output_dir, i)
                total_time += time
                f_sge.write(text)
            elif fid in dev_man_id:
                output_dir = os.path.join(os.path.join(destination, "dev-man"), "audio-{:04}".format(i))
                os.makedirs(output_dir, exist_ok=True)
                text, time = seame_splitText(str(flacFile), output_dir, i)
                total_time += time
                f_man.write(text)
            else:
                output_dir = os.path.join(os.path.join(destination, "train"), "audio-{:04}".format(i))
                os.makedirs(output_dir, exist_ok=True)
                text, time = seame_splitText(str(flacFile), output_dir, i)
                total_time += time
                f_train.write(text)
        verbose_time(args.dataset, total_time)
    elif args.dataset.lower() == "hkust":
        jieba.set_dictionary(os.path.join(BASE_DIR, "jieba_dict/dict.txt.big"))
        dataset_root = "hkust_mcts_p1"
        destination = "splitted-hkust"
        sph_files = sorted(list(Path(dataset_root).rglob("*.sph")))
        print(len(sph_files))
        os.makedirs(destination, exist_ok=True)
        output_text_file = os.path.join(destination, "whole_text.txt")
        f = open(output_text_file, "w")
        total_time = 0.0
        for i, sph_file in enumerate(sph_files):
            output_dir = os.path.join(destination, "audio-{:04}".format(i))
            os.makedirs(output_dir, exist_ok=True)
            text, time = hkust_splitText(str(sph_file), output_dir, i)
            total_time += time
            f.write(text)
        verbose_time(args.dataset, total_time)
    elif args.dataset.lower() == "aishell-2":
        jieba.set_dictionary(os.path.join(BASE_DIR, "jieba_dict/dict.txt.big"))
        transcript_file = "AISHELL-2/iOS/data/trans.txt"
        destination = "AISHELL-2/iOS/data/wav"
        trans = pd.read_csv(transcript_file, sep="\t", header=None)
        output_text_file = os.path.join(destination, "whole_text.txt")
        f = open(output_text_file, "w")
        with Pool(16) as pool:
            processed_text = pool.map(process_aishell_text, trans[1])
        for fn, text in zip(trans[0], processed_text):
            folder = os.path.split(fn)[0]
            os.makedirs(os.path.join(destination, folder), exist_ok=True)
            with open(os.path.join(destination, fn.replace(".wav", ".txt")), "w") as fout:
                fout.write(text + "\n")
            f.write(text + "\n")
    elif args.dataset.lower() == "formosa":
        dataset_root = "NER-Trs-All/Train"
        output_text_file = os.path.join(dataset_root, "whole_text.txt")
        f = open(output_text_file, "w")
        texts = list(Path(dataset_root).rglob("*.txt"))
        for text in texts:
            with open(text, "r") as fin:
                data = fin.read()
            f.write(data)
    elif args.dataset.lower() == "librispeech":
        dataset_root = "LibriSpeech"
        output_text_file = os.path.join(dataset_root, "whole_text.txt")
        f = open(output_text_file, "w")
        train_360 = "LibriSpeech/train-clean-360"
        train_100 = "LibriSpeech/train-clean-100"
        flac_files = list(Path(train_360).rglob("*.flac")) + list(Path(train_100).rglob("*.flac"))
        with Pool(16) as pool:
            texts = pool.map(read_libri_text, flac_files)
        print(len(texts))
        f.write("\n".join(texts))
    elif args.dataset.lower() == "eatmic":
        dataset_root = "EATMIC"
        destination = "split/EATMIC"
        train_set = "EATMIC/set/train.set"
        dev_set = "EATMIC/set/dev.set"
        test_set = "EATMIC/set/test.set"
        male_spk_ids = sorted(os.listdir("EATMIC/wavfile/male"))
        male_id = dict((spk, i) for i, spk in enumerate(male_spk_ids))
        female_spk_ids = sorted(os.listdir("EATMIC/wavfile/female"))
        female_id = dict((spk, i) for i, spk in enumerate(female_spk_ids))
        os.makedirs(destination, exist_ok=True)
        for set_name in ["train", "dev", "test", "other"]:
            os.makedirs(os.path.join(destination, set_name), exist_ok=True)
            os.makedirs(os.path.join(destination, set_name, "male"), exist_ok=True)
            os.makedirs(os.path.join(destination, set_name, "female"), exist_ok=True)
        
        def read_set(set_file):
            with open(set_file, "r") as f:
                fids = f.read().strip("\n").split("\n")
            return fids
        train_ids = read_set(train_set)
        dev_ids = read_set(dev_set)
        test_ids = read_set(test_set)
        wavs = list(Path(dataset_root).rglob("*.wav"))
        commands = []
        mkdir_command = []
        for wav in wavs:
            paths = str(wav).split("/")
            gender = "m" if paths[2] == "male" else "f"
            spk_id = male_id[paths[3]] if gender == "m" else female_id[paths[3]]
            sent_id = int(paths[-1][6:8])
            fid = "eatmic-{}si{:03}-sen{:04}".format(gender, spk_id, sent_id)
            if fid in train_ids:
                if gender == "m":
                    output_dir = os.path.join(destination, "train", "male", paths[3])
                else:
                    output_dir = os.path.join(destination, "train", "female", paths[3])
            elif fid in dev_ids:
                if gender == "m":
                    output_dir = os.path.join(destination, "dev", "male", paths[3])
                else:
                    output_dir = os.path.join(destination, "dev", "female", paths[3])
            elif fid in test_ids:
                if gender == "m":
                    output_dir = os.path.join(destination, "test", "male", paths[3])
                else:
                    output_dir = os.path.join(destination, "test", "female", paths[3])
            else:
                if gender == "m":
                    output_dir = os.path.join(destination, "other", "male", paths[3])
                else:
                    output_dir = os.path.join(destination, "other", "female", paths[3])
            mkdir_command.append(["mkdir", "-p", output_dir])
            commands.append(["cp", str(wav), output_dir])
            commands.append(["cp", str(wav) + ".lab", output_dir])
        mkdir_command = list(set(mkdir_command))
        with Pool(16) as pool:
            pool.map(exec_cmd, mkdir_command)
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
    elif args.dataset.lower() == "split/eatmic":
        dataset_root = "split/EATMIC"
        destination = "split/splitted-coursera"
        commands = []
        set_select = {"train": 20, "dev": 3, "test": 3}
        for set_name in ["train", "dev", "test"]:
            male_spk = os.listdir("split/EATMIC/{}/male".format(set_name))
            female_spk = os.listdir("split/EATMIC/{}/female".format(set_name))
            perturb_method = ["none", "tempo", "speed", "pitch", "denoise"]
            np.random.shuffle(male_spk)
            np.random.shuffle(female_spk)
            select = set_select[set_name]
            for spk in male_spk[:select]:
                methods = set()
                while len(methods) < 2:
                    methods.add(np.random.choice(perturb_method))
                input_dir = os.path.join("split/EATMIC/{}/male".format(set_name), spk)
                output_dir = os.path.join(destination, set_name, spk)
                subprocess.call(["mkdir", "-p", output_dir])
                commands.extend(perturb_eatmic(input_dir, methods, output_dir))
            for spk in female_spk[:select]:
                methods = set()
                while len(methods) < 2:
                    methods.add(np.random.choice(perturb_method))
                input_dir = os.path.join("split/EATMIC/{}/female".format(set_name), spk)
                output_dir = os.path.join(destination, set_name, spk)
                subprocess.call(["mkdir", "-p", output_dir])
                commands.extend(perturb_eatmic(input_dir, methods, output_dir))
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
    elif args.dataset.lower() == "cool-small-eatmic":
        dataset_root = "split/EATMIC"
        destination = "split/cool-small"
        commands = []
        for set_name in ["train", "other"]:
            male_spk = os.listdir("split/EATMIC/{}/male".format(set_name))
            female_spk = os.listdir("split/EATMIC/{}/female".format(set_name))
            perturb_method = ["none", "tempo", "speed", "pitch"]
            np.random.shuffle(male_spk)
            np.random.shuffle(female_spk)
            for spk in male_spk:
                methods = set()
                while len(methods) < 2:
                    methods.add(np.random.choice(perturb_method))
                input_dir = os.path.join("split/EATMIC/{}/male".format(set_name), spk)
                output_dir = os.path.join(destination, "train", "male", spk)
                subprocess.call(["mkdir", "-p", output_dir])
                commands.extend(perturb_eatmic(input_dir, methods, output_dir))
            for spk in female_spk:
                methods = set()
                while len(methods) < 2:
                    methods.add(np.random.choice(perturb_method))
                input_dir = os.path.join("split/EATMIC/{}/female".format(set_name), spk)
                output_dir = os.path.join(destination, "train", "female", spk)
                subprocess.call(["mkdir", "-p", output_dir])
                commands.extend(perturb_eatmic(input_dir, methods, output_dir))
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
    elif args.dataset.lower() == "cool-small-seame":
        dataset_root = "split/splitted-seame"
        destination = "split/cool-small"
        commands = []
        with open("stat/seame-train.json", "r") as f:
            stats = json.load(f)
        audios = [d for d in os.listdir("split/splitted-seame/train/") if not d.endswith(".txt")]
        print(len(audios))
        perturb_method = ["none", "tempo", "speed", "pitch"]
        for audio in audios:
            methods = set()
            while len(methods) < 2:
                methods.add(np.random.choice(perturb_method))
            input_dir = os.path.join("split/splitted-seame/train", audio)
            output_dir = os.path.join(destination, "train", audio)
            subprocess.call(["mkdir", "-p", output_dir])
            commands.extend(perturb_other(input_dir, methods, output_dir, stats))
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
    elif args.dataset.lower() == "cool-small-coursera":
        dataset_root = "split/splitted-coursera"
        destination = "split/cool-small"
        commands = []
        with open("stat/coursera-train.json", "r") as f:
            stats = json.load(f)
        audios = [d for d in os.listdir("split/splitted-coursera/train/") if d.startswith("course")]
        print(len(audios))
        perturb_method = ["none", "tempo", "speed", "pitch"]
        for audio in audios:
            input_dir = os.path.join("split/splitted-coursera/train", audio)
            output_dir = os.path.join(destination, "train", audio)
            subprocess.call(["mkdir", "-p", output_dir])
            methods = np.random.choice(perturb_method)
            commands.extend(perturb_coursera(input_dir, methods, output_dir, stats))
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
    elif args.dataset.lower() == "cool-small-formosa":
        dataset_root = "split/NER-Trs-All"
        destination = "split/cool-small"
        commands = []
        with open("stat/formosa.json", "r") as f:
            stats = json.load(f)
        with open("stat/cool-small-train.json", "r") as f:
            destination_stats = json.load(f)
        wavs = list(Path("split/NER-Trs-All").rglob("*.wav"))
        short_wavs = []
        for wav in wavs:
            fid = str(wav).replace(".wav", "")
            code_rate, audio_length = stats[fid]
            if audio_length > 0.2 and audio_length < 20:
                short_wavs.append(wav)
        print(len(short_wavs))
        perturb_method = ["none", "tempo", "speed", "pitch"]
        output_dir = os.path.join(destination, "train", "formosa")
        subprocess.call(["mkdir", "-p", output_dir])
        for wav in short_wavs:
            fid = str(wav).replace(".wav", "")
            text = read_formosa_text(wav)
            output_wav = os.path.join(output_dir, str(wav).split("/")[-1])
            output_txt = output_wav.replace(".wav", ".txt")
            with open(output_txt, "w") as f:
                f.write(text)
            commands.append(["cp", str(wav), output_wav])
            output_fid = output_wav.replace(".wav", "")
            destination_stats[output_fid] = stats[fid]
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
        with open("stat/cool-small-train.json", "w") as f:
            json.dump(destination_stats, f)
    elif args.dataset.lower() == "cool-small-librispeech":
        dataset_root = "split/LibriSpeech/train-clean-100"
        destination = "split/cool-small"
        commands = []
        with open("stat/librispeech-100.json", "r") as f:
            stats = json.load(f)
        with open("stat/cool-small-train.json", "r") as f:
            destination_stats = json.load(f)
        flacs = list(Path("split/LibriSpeech/train-clean-100").rglob("*.flac"))

        print(len(flacs))
        output_dir = os.path.join(destination, "train", "librispeech")
        subprocess.call(["mkdir", "-p", output_dir])
        for flac in flacs:
            fid = str(flac).replace(".wav", "")
            text = read_librispeech_text(flac)
            output_wav = os.path.join(output_dir, str(flac).replace(".flac", ".wav").split("/")[-1])
            output_txt = output_wav.replace(".wav", ".txt")
            with open(output_txt, "w") as f:
                f.write(text)
            commands.append(["ffmpeg", "-y", "-i", str(flac), output_wav])
            output_fid = output_wav.replace(".wav", "")
            destination_stats[output_fid] = stats[fid]
        with Pool(16) as pool:
            pool.map(exec_cmd, commands)
        with open("stat/cool-small-train.json", "w") as f:
            json.dump(destination_stats, f)
    elif args.dataset.lower() == "cool-small":
        dataset_root = "split/cool-small"
        for set_name in ["train", "dev", "test"]:
            collect_whole_text(os.path.join(dataset_root, set_name))
    elif args.dataset.lower() == "all-text":
        # collect_whole_chinese_text("split")
        collect_whole_text("split/LibriSpeech/train-clean-100")
        collect_whole_text("split/LibriSpeech/train-clean-360")
        collect_whole_text("split/LibriSpeech/dev-clean")
        collect_whole_text("split/LibriSpeech/test-clean")
    elif args.dataset.lower() == "ocw":
        dataset_root = "OCW"
        destination = "split/ocw"
        os.makedirs(destination, exist_ok=True)
        courses = sorted(os.listdir(dataset_root))
        audio_id = 0
        total_time = 0.0
        output_whole_text = open(os.path.join(destination, "whole_text.txt"), "w")
        for i, course in enumerate(courses):
            output_dir = os.path.join(destination, "course-{:02}".format(i))
            
            os.makedirs(output_dir, exist_ok=True)
            root = os.path.join(dataset_root, course)
            srt_files = list(Path(root).rglob("*.srt"))
            vtt_files = list(Path(root).rglob("*.vtt"))
            subtitles = sorted(srt_files + vtt_files)
            subtitles = [str(sub) for sub in subtitles]
            audio_ids = list(range(audio_id, audio_id + 1 + len(subtitles)))
            text = ""
            with Pool(16) as pool:
                outputs = pool.starmap(coursera_splitText, zip(subtitles, [output_dir]*len(subtitles), audio_ids))
            for filetext, filetime in outputs:
                text += filetext
                total_time += filetime
            output_whole_text.write(text)
            audio_id += len(subtitles)
        verbose_time("OCW", total_time)
    elif args.dataset.lower() == "hung-yi":
        dataset_root = "Hung-Yi"
        destination = "split/hung-yi"
        os.makedirs(destination, exist_ok=True)
        courses = sorted(os.listdir(dataset_root))
        audio_id = 0
        total_time = 0.0
        output_whole_text = open(os.path.join(destination, "whole_text.txt"), "w")
        ml_dir = os.path.join(destination, "ml")
        os.makedirs(ml_dir, exist_ok=True)
        mlds_dir = os.path.join(destination, "mlds")
        os.makedirs(mlds_dir, exist_ok=True)

        vtt_files = list(Path(dataset_root).rglob("*.vtt"))
        subtitles = sorted(vtt_files)
        subtitles = [str(sub) for sub in subtitles]
        ml_subs = [sub for sub in subtitles if os.path.split(sub)[-1].startswith("ML")]
        mlds_subs = [sub for sub in subtitles if not os.path.split(sub)[-1].startswith("ML")]
        for directory, subs in [(ml_dir, ml_subs), (mlds_dir, mlds_subs)]:
            audio_ids = list(range(audio_id, audio_id + 1 + len(subs)))
            text = ""
            with Pool(16) as pool:
                outputs = pool.starmap(coursera_splitText, zip(subs, [directory]*len(subs), audio_ids))
            for filetext, filetime in outputs:
                text += filetext
                total_time += filetime
            output_whole_text.write(text)
            audio_id += len(subtitles)
        verbose_time("hung-yi", total_time)
    elif args.dataset.lower() == "hsinmu":
        # dataset_root = "/home/galen/data/hsinmu/1"
        # destination = "/home/galen/data/hsinmu/split/1"
        dataset_root = "/home/galen/data/hsinmu/2"
        destination = "/home/galen/data/hsinmu/split/2"
        os.makedirs(destination, exist_ok=True)
        audio_id = 0
        total_time = 0.0
        output_whole_text = open(os.path.join(destination, "whole_text.txt"),
                "w")

        # srt_files = list(Path(dataset_root).rglob("*.srt"))
        srt_files = list(Path(dataset_root).glob("*.srt"))
        subtitles = sorted(srt_files)
        subtitles = [str(sub) for sub in subtitles]
        for subs in subtitles:
            audio_ids = list(range(audio_id, audio_id + 1 + len(subs)))
            text = ""
            with Pool(16) as pool:
                outputs = pool.starmap(coursera_splitText,
                        zip(subtitles, [destination]*len(subtitles), audio_ids))
            for filetext, filetime in outputs:
                text += filetext
                total_time += filetime
            output_whole_text.write(text)
            audio_id += len(subtitles)
        verbose_time("hsinmu", total_time)

    elif args.dataset.lower() == "new_cool":
        dataset_name = "test"
        dataset_root = "/home/galen/data/5.4_new_data/" + dataset_name
        destination = "/home/galen/data/5.4_new_data/split/" + dataset_name
        os.makedirs(destination, exist_ok=True)
        # courses = sorted(os.listdir(dataset_root))
        audio_id = 0
        total_time = 0.0
        output_whole_text = open(os.path.join(destination, "whole_text.txt"),
                "w")

        srt_files = list(Path(dataset_root).rglob("*.srt"))
        subtitles = sorted(srt_files)
        subtitles = [str(sub) for sub in subtitles]
    # for subs in subtitles:
        audio_ids = list(range(audio_id, audio_id + 1 + len(subtitles)))
        # print(audio_ids)
        text = ""
        # print("zip: {}".format(list(zip(subtitles, [destination]*len(subtitles), audio_ids))))
        # with Pool(16) as pool:
            # outputs = pool.starmap(new_cool_split_text,
                    # zip(subtitles, [destination]*len(subtitles), audio_ids))
        audio_list = zip(subtitles, [destination]*len(subtitles), audio_ids)
        # print(list(audio_list))
        outputs = []
        for audio in audio_list:
            outputs.append(new_cool_split_text(*audio))
        for filetext, filetime in outputs:
            text += filetext
            total_time += filetime
        output_whole_text.write(text)
        audio_id += len(subtitles)
        verbose_time("new_cool", total_time)
    elif args.dataset.lower() == "lecture":
        dataset_root = "Lecture"
        destination = "split/lecture"
        ss_text_dir = "Lecture/LectureCSdata/LectureSS2006/local"
        ss_wav_dir = "Lecture/LectureSS/wavfile"
        dsp_text_dir = "Lecture/LectureCSdata/LectureDSP/local"
        dsp_wav_dir = "Lecture/LectureDSP/wavfile"
        
        os.makedirs(destination, exist_ok=True)
        ss_dir = os.path.join(destination, "ss")
        os.makedirs(ss_dir, exist_ok=True)
        dsp_dir = os.path.join(destination, "dsp")
        os.makedirs(dsp_dir, exist_ok=True)
        def get_dicts(text_dir):
            set_text = ["train.text", "dev.text", "test.text"]
            dicts = []
            test_exist = os.path.exists(os.path.join(text_dir, "test.text"))
            for text in set_text:
                text_file = os.path.join(text_dir, text)
                if not os.path.exists(text_file):
                    continue
                fid_dict = {}
                with open(text_file, "r") as f:
                    data = f.read().strip("\n").split("\n")
                for line in data:
                    line = line.strip(" ").split(" ")
                    fid = line[0]
                    text = " ".join(line[1:])
                    fid_dict[fid] = text
                dicts.append(fid_dict)
            if test_exist:
                return dicts[0], dicts[1], dicts[2]
            else:
                return dicts[0], dicts[1], {}
            
        for text_dir, wav_dir, output_dir in [(ss_text_dir, ss_wav_dir, ss_dir), (dsp_text_dir, dsp_wav_dir, dsp_dir)]:
            train_dir = os.path.join(output_dir, "train")
            dev_dir = os.path.join(output_dir, "dev")
            test_dir = os.path.join(output_dir, "test")
            for directory in [train_dir, dev_dir, test_dir]:
                os.makedirs(directory, exist_ok=True)
            wavs = [str(wav) for wav in list(Path(wav_dir).rglob("*.wav"))]
            train_dict, dev_dict, test_dict = get_dicts(text_dir)
            for wav in wavs:
                fid = os.path.split(wav)[-1].replace(".wav", "")
                if fid in train_dict:
                    subprocess.call(["cp", wav, train_dir])
                    text = train_dict[fid]
                    text_file = os.path.join(train_dir, fid) + ".txt"
                    with open(text_file, "w") as f:
                        f.write(text + "\n")
                elif fid in dev_dict:
                    subprocess.call(["cp", wav, dev_dir])
                    text = dev_dict[fid]
                    text_file = os.path.join(dev_dir, fid) + ".txt"
                    with open(text_file, "w") as f:
                        f.write(text + "\n")
                elif fid in test_dict:
                    subprocess.call(["cp", wav, test_dir])
                    text = test_dict[fid]
                    text_file = os.path.join(test_dir, fid) + ".txt"
                    with open(text_file, "w") as f:
                        f.write(text + "\n")
    elif args.dataset.lower() == "coursera-new":
        dataset_root = "coursera-new"
        destination = "split/splitted-coursera/new"
        os.makedirs(destination, exist_ok=True)
        jieba.set_dictionary(os.path.join(BASE_DIR, "jieba_dict/dict.txt.big"))
        audio_id = 0
        course_id = 0
        output_whole_text = open(os.path.join(destination, "whole_text.txt"), "w")
        set_time = 0.0
        courses = ["the-red-chamber-dream-daguan-garden", "taiwan-medical", "inquiry-into-confucius-and-mencius"]
        for i, course in enumerate(courses):
            audio_dir = os.path.join(dataset_root, course)
            srt_files = list(Path(audio_dir).rglob("*.zh-TW.srt"))
            if len(srt_files) == 0:
                srt_files = list(Path(audio_dir).rglob("*.en.srt"))
            text = ""
            srt_files = [str(srt) for srt in srt_files]
            audio_ids = list(range(audio_id, audio_id + 1 + len(srt_files)))
            output_dir = os.path.join(destination, "course-{:02}".format(i))
            os.makedirs(output_dir, exist_ok=True)
            with Pool(16) as pool:
                outputs = pool.starmap(coursera_splitText, zip(srt_files, [output_dir]*len(srt_files), audio_ids))
            for filetext, filetime in outputs:
                text += filetext
                set_time += filetime
            output_whole_text.write(text)
            audio_id += len(srt_files)
        # verbose_time('new', time)
