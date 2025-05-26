#!/bin/bash
for file in *.mp3 *.m4a *.mp4; do
  ffmpeg -i "$file" "${file%.*}.wav"
done