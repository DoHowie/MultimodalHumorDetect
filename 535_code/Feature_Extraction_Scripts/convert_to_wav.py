import os
from audio_extract import extract_audio

def convert_all_mp4_to_wav(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            mp4_path = os.path.join(input_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"  # Change extension
            wav_path = os.path.join(output_dir, wav_filename)

            # Extract audio and save as WAV
            extract_audio(mp4_path, wav_path,output_format='wav')
            print(f"Converted: {mp4_path} -> {wav_path}")

# Example usage
input_directory = "./dataset/urfunny2_videos"  # Change to your MP4 directory
output_directory = "./dataset/urfunny2_audios"  # Directory to save WAV files

convert_all_mp4_to_wav(input_directory, output_directory)
