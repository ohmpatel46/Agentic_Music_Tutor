"""Scale training module for tempo-based note timing analysis."""

import json
import time
import os
from typing import Dict, List, Tuple, Optional
from config import SCALE_TRAINING_PATH, NOTE_TIMING_TOLERANCE


class ScaleTrainer:
    """Handles scale training with tempo-based timing analysis."""
    
    def __init__(self, scale_data: Dict = None):
        """Initialize the scale trainer with scale data."""
        if scale_data is None:
            self.scale_data = self.load_scale_data()
        else:
            self.scale_data = scale_data
        self.tempo_bpm = self.scale_data.get("default_bpm", 60)
        self.current_note_index = 0
        self.actual_timings = []
        self.note_delays = []  # Store delays for each note transition
        self.note_timestamps = []  # Store exact timestamps when each note was hit
        self.start_time = None
        self.is_training = False
        self.expected_timings = []
        self.calculate_expected_timings()
    
    def load_scale_data(self) -> Dict:
        """Load scale training data from JSON file."""
        try:
            with open(SCALE_TRAINING_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Scale file not found: {SCALE_TRAINING_PATH}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON in scale file: {SCALE_TRAINING_PATH}")
            return {}
    
    def get_available_scales(self) -> List[Dict]:
        """Get all available scales from the scales folder."""
        scales = []
        scales_dir = "scales"
        
        if os.path.exists(scales_dir):
            for filename in os.listdir(scales_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(scales_dir, filename)
                        with open(filepath, 'r') as f:
                            scale_data = json.load(f)
                            scales.append({
                                'filename': filename,
                                'name': scale_data.get('scale_name', filename),
                                'notes': scale_data.get('notes', []),
                                'category': scale_data.get('category', 'unknown'),
                                'difficulty': scale_data.get('difficulty', 'unknown')
                            })
                    except Exception as e:
                        print(f"Error loading scale {filename}: {e}")
        
        return scales
    
    def load_scale_by_name(self, scale_name: str) -> bool:
        """Load a specific scale by name."""
        scales_dir = "scales"
        
        if os.path.exists(scales_dir):
            for filename in os.listdir(scales_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(scales_dir, filename)
                        with open(filepath, 'r') as f:
                            scale_data = json.load(f)
                            if scale_data.get('scale_name') == scale_name:
                                self.scale_data = scale_data
                                self.tempo_bpm = scale_data.get("default_bpm", 60)
                                self.calculate_expected_timings()
                                return True
                    except Exception as e:
                        print(f"Error loading scale {filename}: {e}")
        
        return False
    
    def set_tempo(self, bpm: int):
        """Set the tempo for the scale training."""
        self.tempo_bpm = max(self.scale_data["tempo_range"]["min_bpm"], 
                           min(self.scale_data["tempo_range"]["max_bpm"], bpm))
        self.calculate_expected_timings()
    
    def calculate_expected_timings(self):
        """Calculate expected note timings based on current tempo."""
        # Convert BPM to seconds per beat
        seconds_per_beat = 60.0 / self.tempo_bpm
        
        # For quarter notes, each note gets one beat
        self.expected_timings = []
        for i in range(len(self.scale_data["notes"])):
            expected_time = i * seconds_per_beat
            self.expected_timings.append(expected_time)
    
    def start_training(self):
        """Start a new scale training session."""
        self.current_note_index = 0
        self.actual_timings = []
        self.note_delays = []  # Reset note delays
        self.note_timestamps = [] # Reset note timestamps
        self.start_time = None  # Will be set when first note is detected
        self.is_training = True
        self.calculate_expected_timings()
    
    def stop_training(self):
        """Stop the current training session."""
        self.is_training = False
        self.start_time = None
    
    def check_note_timing(self, detected_note: str) -> Tuple[bool, float, float]:
        """Check if a detected note matches the expected note and timing."""
        if not self.is_training:
            return False, 0.0, 0.0
        
        expected_note = self.scale_data["notes"][self.current_note_index]
        
        # Check if note matches
        if detected_note == expected_note:
            current_time = time.time()
            
            # Set start time when first note is detected
            if self.start_time is None:
                self.start_time = current_time
            
            # Calculate delay instantly when note is detected
            if self.current_note_index > 0:  # Not the first note
                # Get the timestamp of the previous note
                previous_note_timestamp = self.note_timestamps[-1]
                
                # Calculate expected interval between notes
                expected_interval = 60.0 / self.tempo_bpm  # e.g., 1.0s at 60 BPM
                
                # Calculate actual interval since previous note
                actual_interval = current_time - previous_note_timestamp
                
                # Calculate delay (difference between actual and expected interval)
                timing_error = actual_interval - expected_interval
                
                # Record the delay immediately
                self.note_delays.append(timing_error)
                
                # Record actual timing (total time since start)
                self.actual_timings.append(current_time - self.start_time)
                
                # Record exact timestamp for the note
                self.note_timestamps.append(current_time)
                
                # Move to next note
                self.current_note_index += 1
                
                return True, timing_error, expected_interval
            else:
                # First note - just record the timing and move to next
                self.actual_timings.append(0.0)  # First note is reference point
                self.note_timestamps.append(current_time) # Record timestamp for the first note
                self.note_delays.append(0.0)  # First note has no delay
                self.current_note_index += 1
                return True, 0.0, 0.0
        
        return False, 0.0, 0.0
    
    def get_training_progress(self) -> Dict:
        """Get current training progress and statistics."""
        if not self.is_training:
            return {"status": "not_started"}
        
        total_notes = len(self.scale_data["notes"])
        completed_notes = self.current_note_index
        
        # Calculate average delay (absolute values, excluding first note)
        avg_delay = 0.0
        if len(self.note_delays) > 1:  # Need at least 2 notes to calculate delays
            # Take absolute values of all delays and average them
            absolute_delays = [abs(delay) for delay in self.note_delays[1:]]  # Skip first note (index 0)
            if absolute_delays:
                avg_delay = sum(absolute_delays) / len(absolute_delays)
        
        return {
            "status": "training",
            "scale_name": self.scale_data["scale_name"],
            "tempo_bpm": self.tempo_bpm,
            "current_note_index": self.current_note_index,
            "total_notes": total_notes,
            "completed_notes": completed_notes,
            "progress_percentage": (completed_notes / total_notes) * 100,
            "average_delay": avg_delay,
            "next_expected_note": self.scale_data["notes"][self.current_note_index] if self.current_note_index < total_notes else None,
            "next_expected_time": self.expected_timings[self.current_note_index] if self.current_note_index < total_notes else None
        }
    
    def get_expected_note(self) -> str:
        """Get the next expected note in the scale."""
        if self.current_note_index < len(self.scale_data["notes"]):
            return self.scale_data["notes"][self.current_note_index]
        return None
    
    def get_scale_info(self) -> Dict:
        """Get information about the current scale."""
        return {
            "name": self.scale_data["scale_name"],
            "notes": self.scale_data["notes"],
            "tempo_range": self.scale_data["tempo_range"]
        }
