"""Audio processing utilities for the Agentic Music Tutor."""

import numpy as np
import sounddevice as sd
from src.data.config import SAMPLE_RATE, CHANNELS
from typing import Optional


def record_audio_chunk(duration: float = 0.5) -> Optional[np.ndarray]:
    """Record a short audio chunk for real-time processing."""
    try:
        # Record a short chunk
        audio_chunk = sd.rec(
            int(SAMPLE_RATE * duration),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        # Convert to mono if stereo
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        return audio_chunk
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None


def get_audio_stats(audio_chunk: np.ndarray) -> dict:
    """Get statistics about an audio chunk for debugging."""
    if audio_chunk is None or len(audio_chunk) == 0:
        return {}
    
    return {
        "samples": len(audio_chunk),
        "min_amplitude": float(audio_chunk.min()),
        "max_amplitude": float(audio_chunk.max()),
        "rms": float(np.sqrt(np.mean(audio_chunk**2))),
        "peak": float(np.max(np.abs(audio_chunk)))
    }
