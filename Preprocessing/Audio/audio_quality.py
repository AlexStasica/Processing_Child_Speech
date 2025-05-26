import librosa
import numpy as np
import glob
import argparse


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def main(data_dir, output_file):

    audio_files = sorted(glob.glob(f'{data_dir}/*.wav'))
    print(audio_files)

    # collect signal-to-noise ratios for all files in the directory
    snrs = []

    with open(output_file, 'w') as outfile:

        for audio in audio_files:
            y, sr = librosa.load(audio)
            snr = signaltonoise(y)
            formatted_snr = format(snr, '.8f')
            snrs.append(snr)
            print(f"{formatted_snr}")  # print to console
            outfile.write(audio + '\t' + formatted_snr + '\n')

    # compute the mean and std
    avg_snr = np.mean(snrs)
    std_snr = np.std(snrs)

    # report summary
    result_summary = f"Directory: {data_dir}\nAverage SNR: {avg_snr:.8f}\nStandard Deviation: {std_snr:.8f}\n"
    print(result_summary.strip())  # print to console


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Signal-to-Noise Ratio for audio files")
    parser.add_argument('data_dir', type=str, help="Directory containing audio files")
    parser.add_argument('output_file', type=str, help="Path to the output txt file")
    args = parser.parse_args()
    main(args.data_dir, args.output_file)