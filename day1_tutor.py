import argparse
import json
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional at runtime
    sd = None  # type: ignore

try:
    import aubio  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("aubio is required. Please install with `pip install aubio`." ) from exc


NOTE_NAMES_SHARP = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


def load_ground_truth(ground_truth_path: str) -> Tuple[str, List[str]]:
    """Load ground truth JSON with fields: song, notes."""
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    song_name = str(data.get("song", "Unknown_Song"))
    expected_notes = [str(n) for n in data.get("notes", [])]
    return song_name, expected_notes


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Return a mono float32 signal; average channels if input is stereo."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    mono = np.mean(audio, axis=1)
    return mono.astype(np.float32)


def record_audio(duration_seconds: float, sample_rate_hz: int, channels: int = 1) -> np.ndarray:
    """Record audio from the default system microphone and return mono float32."""
    if sd is None:
        raise SystemExit("sounddevice is required for recording. Install with `pip install sounddevice`.")
    
    num_frames = int(duration_seconds * sample_rate_hz)
    print(f"Recording {duration_seconds}s @ {sample_rate_hz} Hz, ch={channels}")
    
    audio = sd.rec(frames=num_frames, samplerate=sample_rate_hz, channels=channels, dtype="float32")
    sd.wait()
    
    mono = ensure_mono(audio)
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    print(f"Captured — RMS: {rms:.4f}, Peak: {peak:.4f}")
    
    if peak < 1e-4:
        print("Warning: Captured near-silence. Check microphone permissions.")
    
    return mono


def read_wav_file(wav_path: str) -> Tuple[int, np.ndarray]:
    """Read a WAV file and return (sample_rate, mono float32 audio in [-1,1])."""
    sample_rate, data = wavfile.read(wav_path)
    # Normalize to float32 in [-1, 1]
    if data.dtype == np.int16:
        audio = (data.astype(np.float32)) / 32768.0
    elif data.dtype == np.int32:
        audio = (data.astype(np.float32)) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = ((data.astype(np.float32)) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    return sample_rate, ensure_mono(audio)


def save_wav_file(wav_path: str, sample_rate_hz: int, audio: np.ndarray) -> None:
    """Save mono float32 audio to WAV as int16."""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)
    wavfile.write(wav_path, sample_rate_hz, int16_audio)


def hz_to_midi(hz: float) -> Optional[int]:
    """Convert frequency in Hz to nearest MIDI note number, or None if invalid."""
    if hz is None or hz <= 0:
        return None
    midi_float = 69.0 + 12.0 * math.log2(hz / 440.0)
    return int(round(midi_float))


def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI number to note name like 'A4'."""
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES_SHARP[midi_note % 12]
    return f"{name}{octave}"


def detect_note_sequence(audio: np.ndarray, sample_rate_hz: int, *, hop_size: int = 256, buffer_size: int = 1024) -> List[str]:
    """Detect a monophonic sequence of note names from raw audio using aubio - optimized for guitar."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    pitch_detector = aubio.pitch("yin", buffer_size, hop_size, sample_rate_hz)
    pitch_detector.set_unit("Hz")

    num_frames_total = audio.shape[0]
    frame_notes: List[Optional[str]] = []

    # Iterate over hop-sized frames
    for start in range(0, num_frames_total - hop_size + 1, hop_size):
        frame = audio[start : start + hop_size]
        
        pitch_hz = float(pitch_detector(frame)[0])
        confidence = float(pitch_detector.get_confidence())
        
        # Much more selective - only accept strong, confident pitches
        if confidence > 0.9:  # High confidence threshold
            midi_note = hz_to_midi(pitch_hz)
            if midi_note is not None and 0 <= midi_note <= 127:
                frame_notes.append(midi_to_note_name(midi_note))
            else:
                frame_notes.append(None)
        else:
            frame_notes.append(None)

    # Compress consecutive frames into note sequence
    compressed_notes: List[str] = []
    if not frame_notes:
        return compressed_notes

    current_note: Optional[str] = frame_notes[0]
    run_length: int = 1

    def maybe_commit(note: Optional[str], length: int) -> None:
        if note is None:
            return
        if length >= 10:  # Only accept notes that last at least 10 frames (more sustained)
            if len(compressed_notes) == 0 or compressed_notes[-1] != note:
                compressed_notes.append(note)

    for f_note in frame_notes[1:]:
        if f_note == current_note:
            run_length += 1
        else:
            maybe_commit(current_note, run_length)
            current_note = f_note
            run_length = 1
    # Commit last run
    maybe_commit(current_note, run_length)

    return compressed_notes


def compare_sequences(detected: List[str], expected: List[str]) -> float:
    """Print detected vs expected note-by-note and return accuracy percentage."""
    total = max(len(expected), 1)
    correct = 0
    for idx in range(len(expected)):
        expected_note = expected[idx]
        detected_note = detected[idx] if idx < len(detected) else "(none)"
        is_correct = detected_note == expected_note
        mark = "✅" if is_correct else "❌"
        print(f"Detected: {detected_note} | Expected: {expected_note} {mark}")
        if is_correct:
            correct += 1
    accuracy_pct = 100.0 * (correct / total)
    print(f"Accuracy: {accuracy_pct:.0f}%")
    return accuracy_pct


def main() -> None:
    """CLI: record from default mic or analyze a WAV, then compare to ground truth."""
    parser = argparse.ArgumentParser(
        description="Day 1 — Barebones Note Detection: record from mic or analyze a WAV, detect monophonic notes with aubio, and compare to ground truth."
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Path to a .wav file to analyze. If omitted, records from microphone.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Recording duration in seconds when using microphone mode.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate in Hz for recording and analysis (default: 44100).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of input channels when recording (default: 1).",
    )
    # does hop size mean skip frames?
    parser.add_argument(
        "--hop-size",
        type=int,
        default=256,
        help="Hop size in samples for aubio pitch detection (default: 256 - guitar optimized).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1024,
        help="Buffer size in samples for aubio pitch detection (default: 1024 - guitar optimized).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="ground_truth.json",
        help="Path to ground truth JSON file (default: ground_truth.json).",
    )
    parser.add_argument(
        "--save-wav",
        type=str,
        default=None,
        help="Optional path to save recorded audio as WAV.",
    )

    args = parser.parse_args()

    song_name, expected_notes = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for: {song_name}")
    print(f"Expected notes: {expected_notes}")

    if args.wav:
        sample_rate_hz, audio = read_wav_file(args.wav)
        if args.sr and args.sr != sample_rate_hz:
            print(f"Warning: WAV sample rate {sample_rate_hz} differs from --sr {args.sr}. Using {sample_rate_hz}.")
    else:
        if sd is None:
            raise SystemExit("sounddevice is not available. Provide --wav PATH or install sounddevice.")
        print(f"Recording from microphone for {args.duration:.1f}s at {args.sr} Hz. Play the expected riff.")
        audio = record_audio(args.duration, args.sr, channels=args.channels)
        sample_rate_hz = args.sr
        if args.save_wav:
            save_wav_file(args.save_wav, sample_rate_hz, audio)
            print(f"Saved recording to: {args.save_wav}")

    detected_notes = detect_note_sequence(
        audio,
        sample_rate_hz,
        hop_size=args.hop_size,
        buffer_size=args.buffer_size,
    )

    if not detected_notes:
        print("No notes detected. Try playing louder or check microphone.")
        return

    print(f"Detected sequence: {detected_notes}")
    compare_sequences(detected_notes, expected_notes)


if __name__ == "__main__":
    main()


