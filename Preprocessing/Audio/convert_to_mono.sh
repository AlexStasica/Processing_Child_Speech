#!/bin/bash

# Loop through all .wav files in the current directory
for file in *.wav; do
  # Get the filename without the extension
  filename=$(basename "$file" .wav)
  
  # Use sox to convert each file to mono (single channel) and save the output with "_mono" suffix
  sox "$file" "${filename}_mono.wav" channels 1
done

echo "All files have been converted to mono."
