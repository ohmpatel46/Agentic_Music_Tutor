import numpy as np
from scipy.io import wavfile
import sounddevice as sd

SR = 44100
DUR = 5.0
CH = 1

print(f"Recording {DUR}s @ {SR} Hz, ch={CH}, device=default")
audio = sd.rec(int(SR * DUR), samplerate=SR, channels=CH, dtype="float32")
sd.wait()

mono = audio.flatten() if audio.ndim == 2 and audio.shape[1] == 1 else (audio.mean(axis=1) if audio.ndim == 2 else audio)
rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
peak = float(np.max(np.abs(mono))) if mono.size else 0.0
print(f"Captured — RMS: {rms:.4f}, Peak: {peak:.4f}")

wavfile.write("mic_test.wav", SR, (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16))
print("Saved mic_test.wav")