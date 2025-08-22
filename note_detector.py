"""Note detection module using aubio pitch detection algorithms and CREPE."""

import numpy as np
import aubio
import crepe
from typing import Optional, Tuple
from config import BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE
from utils import hz_to_midi, midi_to_note_name


def detect_note_with_crepe(audio_frame: np.ndarray, confidence_threshold: float = 0.3) -> Optional[Tuple[str, str, float]]:
    """Detect note using Google's CREPE model."""
    try:
        # CREPE expects 16kHz sample rate, so we need to resample if needed
        if SAMPLE_RATE != 16000:
            # Simple downsampling for now (you could use librosa.resample for better quality)
            downsample_factor = SAMPLE_RATE // 16000
            audio_16k = audio_frame[::downsample_factor]
        else:
            audio_16k = audio_frame
        
        # Ensure audio is long enough for CREPE (minimum 1024 samples)
        if len(audio_16k) < 1024:
            return None
            
        # Run CREPE prediction
        time, frequency, confidence, activation = crepe.predict(
            audio_16k, 
            sr=16000, 
            step_size=int(16000 * 0.01),  # 10ms step size
            verbose=0
        )
        
        # Get the most confident prediction
        if len(confidence) > 0:
            max_conf_idx = np.argmax(confidence)
            max_confidence = float(confidence[max_conf_idx])
            max_frequency = float(frequency[max_conf_idx])
            
            if max_confidence > confidence_threshold and max_frequency > 0:
                midi_note = hz_to_midi(max_frequency)
                if midi_note is not None and 0 <= midi_note <= 127:
                    note_name = midi_to_note_name(midi_note)
                    return note_name, "CREPE", max_confidence
        
        return None
    except Exception as e:
        print(f"CREPE error: {e}")
        return None


def detect_note_from_frame(
    audio_frame: np.ndarray, 
    confidence_threshold: float = 0.3, 
    algorithm_mode: str = "Multi-Algorithm (Best)"
) -> Optional[Tuple[str, str, float]]:
    """Detect a single note from an audio frame using aubio with algorithm selection and CREPE."""
    if audio_frame.dtype != np.float32:
        audio_frame = audio_frame.astype(np.float32)
    
    # Map algorithm mode to specific algorithms
    if algorithm_mode == "CREPE Only":
        return detect_note_with_crepe(audio_frame, confidence_threshold)
    elif algorithm_mode == "Yin Only":
        algorithms = ["yin"]
    elif algorithm_mode == "YinFFT Only":
        algorithms = ["yinfft"]
    elif algorithm_mode == "MComb Only":
        algorithms = ["mcomb"]
    elif algorithm_mode == "Schmitt Only":
        algorithms = ["schmitt"]
    else:  # Multi-Algorithm (Best) - now includes CREPE
        algorithms = ["yin", "yinfft", "mcomb", "schmitt"]
    
    best_confidence = 0
    best_note = None
    best_algorithm = None
    
    # First try CREPE for multi-algorithm mode
    if algorithm_mode == "Multi-Algorithm (Best)":
        crepe_result = detect_note_with_crepe(audio_frame, confidence_threshold)
        if crepe_result:
            best_note, best_algorithm, best_confidence = crepe_result
    
    # Then try aubio algorithms
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
