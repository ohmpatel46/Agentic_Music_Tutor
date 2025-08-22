"""Audio processing utilities for the Agentic Music Tutor."""

import numpy as np
import sounddevice as sd
from src.data.config import SAMPLE_RATE, CHANNELS
from typing import Optional, Dict
import librosa


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


def analyze_waveform_quality(audio_chunk: np.ndarray) -> Dict:
    """Analyze waveform characteristics for quality assessment."""
    if audio_chunk is None or len(audio_chunk) == 0:
        return {}
    
    try:
        # Basic amplitude statistics
        amplitude_range = np.max(audio_chunk) - np.min(audio_chunk)
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        
        # Normalize audio for analysis
        if amplitude_range > 0:
            normalized_audio = audio_chunk / np.max(np.abs(audio_chunk))
        else:
            normalized_audio = audio_chunk
        
        # Calculate sustain duration (how long note rings out)
        sustain_duration = calculate_sustain_duration(normalized_audio)
        
        # Calculate attack quality (sharpness of note beginning)
        attack_quality = calculate_attack_quality(normalized_audio)
        
        # Calculate frequency stability
        frequency_stability = calculate_frequency_stability(audio_chunk)
        
        # Calculate noise level
        noise_level = calculate_noise_level(normalized_audio)
        
        return {
            "sustain_duration": sustain_duration,
            "attack_quality": attack_quality,
            "dynamic_range": float(amplitude_range),
            "rms_energy": float(rms_energy),
            "frequency_stability": frequency_stability,
            "noise_level": noise_level,
            "overall_quality": calculate_overall_quality(sustain_duration, attack_quality, frequency_stability, noise_level)
        }
    except Exception as e:
        print(f"Error analyzing waveform: {e}")
        return {}


def calculate_sustain_duration(audio: np.ndarray, threshold: float = 0.1) -> float:
    """Calculate how long the note sustains above threshold."""
    try:
        # Find where amplitude drops below threshold
        above_threshold = np.abs(audio) > threshold
        if not np.any(above_threshold):
            return 0.0
        
        # Find the last point above threshold
        last_above = np.where(above_threshold)[0][-1]
        
        # Convert to duration in seconds
        sustain_duration = last_above / SAMPLE_RATE
        return float(sustain_duration)
    except:
        return 0.0


def calculate_attack_quality(audio: np.ndarray, window_size: int = 512) -> float:
    """Calculate the sharpness/quality of the note attack."""
    try:
        if len(audio) < window_size:
            return 0.0
        
        # Look at the first window for attack characteristics
        attack_window = audio[:window_size]
        
        # Calculate rate of amplitude increase
        amplitude_gradient = np.gradient(np.abs(attack_window))
        
        # Higher gradient = sharper attack
        attack_sharpness = np.mean(np.abs(amplitude_gradient))
        
        # Normalize to 0-1 scale
        normalized_sharpness = min(attack_sharpness * 100, 1.0)
        return float(normalized_sharpness)
    except:
        return 0.0


def calculate_frequency_stability(audio: np.ndarray) -> float:
    """Calculate how stable the frequency is throughout the note."""
    try:
        if len(audio) < 1024:
            return 0.0
        
        # Use librosa to get pitch track
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE, hop_length=256)
        
        if len(pitches) == 0:
            return 0.0
        
        # Get the most prominent pitch at each time step
        pitch_track = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Valid pitch
                pitch_track.append(pitch)
        
        if len(pitch_track) < 2:
            return 0.0
        
        # Calculate pitch variation (lower = more stable)
        pitch_std = np.std(pitch_track)
        pitch_mean = np.mean(pitch_track)
        
        if pitch_mean > 0:
            # Coefficient of variation (lower = more stable)
            stability = 1.0 / (1.0 + (pitch_std / pitch_mean))
        else:
            stability = 0.0
        
        return float(stability)
    except:
        return 0.0


def calculate_noise_level(audio: np.ndarray) -> float:
    """Calculate the level of noise in the audio."""
    try:
        if len(audio) < 1024:
            return 0.0
        
        # Calculate signal-to-noise ratio approximation
        # Higher values = less noise
        signal_power = np.mean(audio**2)
        
        # Estimate noise from high-frequency components
        # Simple approach: look at rapid amplitude changes
        amplitude_changes = np.diff(np.abs(audio))
        noise_estimate = np.mean(np.abs(amplitude_changes))
        
        if signal_power > 0 and noise_estimate > 0:
            # Convert to 0-1 scale where 1 = no noise
            snr_ratio = signal_power / (signal_power + noise_estimate)
            return float(snr_ratio)
        else:
            return 0.0
    except:
        return 0.0


def calculate_overall_quality(sustain_duration: float, attack_quality: float, 
                            frequency_stability: float, noise_level: float) -> float:
    """Calculate overall audio quality score from individual metrics."""
    try:
        # Weight the different factors
        # Sustain duration: 25%, Attack: 25%, Frequency stability: 30%, Noise: 20%
        overall_score = (
            sustain_duration * 0.25 +
            attack_quality * 0.25 +
            frequency_stability * 0.30 +
            noise_level * 0.20
        )
        
        return float(overall_score)
    except:
        return 0.0
