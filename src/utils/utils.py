"""Utility functions for the Agentic Music Tutor."""

import json
import numpy as np
from typing import List, Optional, Tuple
import streamlit as st
from src.data.config import NOTE_NAMES_SHARP, GROUND_TRUTH_PATH


def load_ground_truth(ground_truth_path: str = GROUND_TRUTH_PATH) -> Tuple[str, List[str]]:
    """Load ground truth JSON with fields: song, notes."""
    try:
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        song_name = str(data.get("song", "Unknown_Song"))
        expected_notes = [str(n) for n in data.get("notes", [])]
        return song_name, expected_notes
    except Exception as e:
        st.error(f"Error loading ground truth: {e}")
        return "Test_Riff", ["E4", "G4", "A4", "C5"]


def hz_to_midi(hz: float) -> Optional[int]:
    """Convert frequency in Hz to nearest MIDI note number."""
    if hz is None or hz <= 0:
        return None
    midi_float = 69.0 + 12.0 * np.log2(hz / 440.0)
    return int(round(midi_float))


def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI number to note name like 'A4'."""
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES_SHARP[midi_note % 12]
    return f"{name}{octave}"


def calculate_accuracy(detected_notes: List[str], expected_notes: List[str]) -> Tuple[int, int, float]:
    """Calculate accuracy statistics for detected vs expected notes."""
    total_detected = len(detected_notes)
    correct_notes = sum(1 for note in detected_notes if note in expected_notes)
    accuracy = (correct_notes / total_detected * 100) if total_detected > 0 else 0
    return total_detected, correct_notes, accuracy
