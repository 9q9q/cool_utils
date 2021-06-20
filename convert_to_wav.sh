# !/bin/bash

# convert all spaces to _ 
cd $1
rename "s/ /_/g" *

for video in `ls`
do
  ffmpeg -i $video -c copy -map 0:a $video.mp4
  ffmpeg -i $video.mp4 $video.wav
done

rename "s/.mp4.wav/.wav/g" *
