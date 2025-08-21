"""Note detection module using aubio pitch detection algorithms."""

import numpy as np
import aubio
from typing import Optional, Tuple
from config import BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE
from utils import hz_to_midi, midi_to_note_name


def detect_note_from_frame(
    audio_frame: np.ndarray, 
    confidence_threshold: float = 0.3, 
    algorithm_mode: str = "Multi-Algorithm (Best)"
) -> Optional[Tuple[str, str, float]]:
    """Detect a single note from an audio frame using aubio with algorithm selection."""
    if audio_frame.dtype != np.float32:
        audio_frame = audio_frame.astype(np.float32)
    
    # Map algorithm mode to specific algorithms
    if algorithm_mode == "Yin Only":
        algorithms = ["yin"]
    elif algorithm_mode == "YinFFT Only":
        algorithms = ["yinfft"]
    elif algorithm_mode == "MComb Only":
        algorithms = ["mcomb"]
    elif algorithm_mode == "Schmitt Only":
        algorithms = ["schmitt"]
    else:  # Multi-Algorithm (Best)
        algorithms = ["yin", "yinfft", "mcomb", "schmitt"]
    
    best_confidence = 0
    best_note = None
    best_algorithm = None
    
    for algorithm in algorithms:
        try:
            pitch_detector = aubio.pitch(algorithm, BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
            pitch_detector.set_unit("Hz")
            
            # Process the audio frame in BUFFER_SIZE chunks
            for i in range(0, len(audio_frame) - BUFFER_SIZE + 1, HOP_SIZE):
                chunk = audio_frame[i:i + BUFFER_SIZE]
                if len(chunk) == BUFFER_SIZE:
                    pitch_hz = float(pitch_detector(chunk)[0])
                    confidence = float(pitch_detector.get_confidence())
                    
                    # Track the best detection across all algorithms
                    if confidence > best_confidence and confidence > confidence_threshold:
                        midi_note = hz_to_midi(pitch_hz)
                        if midi_note is not None and 0 <= midi_note <= 127:
                            best_confidence = confidence
                            best_note = midi_to_note_name(midi_note)
                            best_algorithm = algorithm
        except Exception as e:
            continue  # Skip algorithms that fail
    
    if best_note:
        return best_note, best_algorithm, best_confidence
    return None
