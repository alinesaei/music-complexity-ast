import os
import librosa
import soundfile as sf
import numpy as np


RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"
TARGET_SR = 16000  # AST standard
DURATION = 10.24  # Exactly 10.24 seconds


def process_audio_smart():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    valid_extensions = ('.mp3', '.wav', '.flac')
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(files)} files. Starting Extraction...")
    target_samples = int(TARGET_SR * DURATION)

    for i, filename in enumerate(files):
        raw_path = os.path.join(RAW_DIR, filename)
        clean_name = os.path.splitext(filename)[0] + "_processed.wav"
        processed_path = os.path.join(PROCESSED_DIR, clean_name)

        try:
            # LOAD THE FULL SONG
            # 16kHz it is very lightweight
            y, sr = librosa.load(raw_path, sr=TARGET_SR, mono=True)

            # Pad if the track is bizarrely short (< 10 seconds)
            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')

            # Scan the song (sliding window approach)
            num_windows = len(y) // target_samples
            max_energy = -1
            best_chunk = None

            # Loop through every 10.24s block in the song
            for w in range(num_windows):
                start = w * target_samples
                end = start + target_samples
                chunk = y[start:end]

                # Calculate the average RMS energy of this specific chunk
                energy = np.mean(librosa.feature.rms(y=chunk))

                # save the most energetic part of the song
                if energy > max_energy:
                    max_energy = energy
                    best_chunk = chunk

            # Normalize
            # Scale the best chunk so its peak volume is exactly 1.0
            y_normalized = librosa.util.normalize(best_chunk)

            sf.write(processed_path, y_normalized, TARGET_SR)
            print(f"[{i + 1}/{len(files)}] Successfuly extracted for {clean_name}")

        except Exception as e:
            print(f"[{i + 1}/{len(files)}] ERROR processing {filename}: {e}")

    print("✅Preprocessing Complete! All 'Drops' are in data/processed/")


if __name__ == "__main__":
    process_audio_smart()