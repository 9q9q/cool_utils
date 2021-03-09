# !/bin/bash
# Usage: prep_dirs.sh /path/to/desired/corpus/dir.

mkdir $1
cd $1 
ln -s $KALDI_ROOT/egs/wsj/s5/steps $1
ln -s $KALDI_ROOT/egs/wsj/s5/utils $1
ln -s $KALDI_ROOT/src $1
                    
cp $KALDI_ROOT/egs/wsj/s5/path.sh $1
sed '1d' path.sh > tmpfile; mv tmpfile path.sh  # KALDI_ROOT already set, remove first line

mkdir exp
mkdir conf
mkdir data
                    
cd $1/data
mkdir train
mkdir lang
mkdir local
                    
cd local
mkdir lang

cd $1
echo -e export train_cmd=\"run.pl\"\\nexport decode_cmd=\"run.pl  --mem 2G\" > cmd.sh
chmod +x cmd.sh
. ./cmd.sh

echo -e --use-energy=false\\n--sample-frequency=16000\\n--allow_downsample=true > conf/mfcc.conf

#echo -e "# !/bin/bash"\\n\\nexport ORIGINAL_BASE=/home/b06901141/video-subtitle-generator--b06901141\\nexport LANG=data/local/lang\\nexport TRAIN=data/train\\nexport LEXICON=data/local/lang/lexicon.txt\\nexport TEXT=data/train/text\\nexport PATH=$PATH:/home/galen/anaconda3/envs/kaldi/bin:/home/galen/learn_kaldi/kaldi/src/lmbin:/home/galen/learn_kaldi/kaldi/src/fstbin\\n\. \./cmd.sh > setup_env.sh
echo "export ORIGINAL_BASE=\"/home/b06901141/video-subtitle-generator--b06901141\"
export LANG=\"data/lang\"
export LOCAL_LANG=\"data/local/lang\"
export TRAIN=\"data/train\"
export LEXICON=\"data/lang/lexicon\"
export TEXT=\"data/train/text\"
export LM=\"/home/galen/coursera_kaldi/lm/lm.arpa.txt\"

export PATH=\$PATH:/home/galen/anaconda3/envs/kaldi/bin:/home/galen/learn_kaldi/kaldi/src/lmbin:/home/galen/learn_kaldi/kaldi/src/fstbin

. ./cmd.sh
" > setup_env.sh
chmod +x setup_env.sh

source setup_env.sh
