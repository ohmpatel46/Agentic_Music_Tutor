import argparse
import numpy as np
from scipy.io import wavfile
import os


def convert_dat_to_wav(dat_path: str, output_path: str = None) -> None:
    """Convert Samsung voice recorder .dat file to WAV format."""
    
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Input file not found: {dat_path}")
    
    if output_path is None:
        base_name = os.path.splitext(dat_path)[0]
        output_path = f"{base_name}.wav"
    
    print(f"Converting {dat_path} to {output_path}")
    
    # Read the .dat file
    with open(dat_path, 'rb') as f:
        raw_data = f.read()
    
    # Samsung voice recorder typically has a header, skip it
    # Try different header sizes
    header_sizes = [0, 44, 128, 256, 512]
    
    for header_size in header_sizes:
        try:
            # Skip header and read audio data
            audio_data = raw_data[header_size:]
            
            # Try to decode as 16-bit PCM
            if len(audio_data) % 2 == 0:
                samples = np.frombuffer(audio_data, dtype=np.int16)
                
                # Check if the data looks reasonable (not all zeros or extreme values)
                if np.any(samples != 0) and np.max(np.abs(samples)) < 32000:
                    print(f"Found valid audio data starting at byte {header_size}")
                    print(f"Audio data: {len(samples)} samples")
                    
                    # Convert to float32 in [-1, 1] range
                    audio_float = samples.astype(np.float32) / 32768.0
                    
                    # Save as WAV with your known settings
                    wavfile.write(output_path, 48000, audio_float)
                    
                    duration = len(audio_float) / 48000
                    print(f"Conversion complete!")
                    print(f"Duration: {duration:.2f} seconds")
                    print(f"Output: {output_path}")
                    return
                    
        except Exception as e:
            continue
    
    # If we get here, try treating as raw PCM with known settings
    print("Trying raw PCM conversion...")
    try:
        # Your settings: 48kHz, mono, 16-bit
        audio_data = raw_data
        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]
        
        samples = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = samples.astype(np.float32) / 32768.0
        
        wavfile.write(output_path, 48000, audio_float)
        
        duration = len(audio_float) / 48000
        print(f"Raw PCM conversion complete!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output: {output_path}")
        
    except Exception as e:
        print(f"All conversion methods failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert Samsung voice recorder .dat files to WAV format")
    parser.add_argument("input", help="Input .dat file path")
    parser.add_argument("-o", "--output", help="Output .wav file path (optional)")
    
    args = parser.parse_args()
    
    try:
        convert_dat_to_wav(
            dat_path=args.input,
            output_path=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
