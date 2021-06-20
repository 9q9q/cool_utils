#!/usr/bin/env bash
# run from dataset/lm dir, e.g. coursera_kaldi/lm

srilm_bin=$HOME/anaconda3/srilm/bin/i686-m64

lm=$1 #e.g. /home/galen/coursera_kaldi/lm/lm.arpa.txt, baseline LM
mix_lm=$2 # e.g. /home/galen/coursera_kaldi/lm/adapt_course00/lm.arpa.txt, LM from just adaptation train set
test_text=$3 # e.g. course00_test.txt, the text you want to evaluate perplexity over

mkdir -p srilm_interp
for w in 0.9 0.8 0.7 0.6 0.5; do
    $srilm_bin/ngram -lm $lm -mix-lm $mix_lm -lambda $w -write-lm srilm_interp/lm.${w}.gz
    echo -n "srilm_interp/lm.${w}.gz "
    $srilm_bin/ngram -lm srilm_interp/lm.${w}.gz -ppl $test_text | paste -s -
done | sort  -k15,15g  > srilm_interp/perplexities.txt
