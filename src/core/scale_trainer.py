"""Scale training module for tempo-based note timing analysis."""

import json
import time
import os
from typing import Dict, List, Tuple, Optional
from src.data.config import SCALE_TRAINING_PATH, NOTE_TIMING_TOLERANCE
import numpy as np


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
        
        # New fields for enhanced analysis
        self.confidence_scores = []      # Track confidence for each note
        self.waveform_metrics = []       # Track audio quality metrics
        self.algorithm_used = []         # Which algorithm detected each note
        self.audio_chunks = []           # Store audio data for each note
        
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
        scales_dir = "src/data/scales"
        
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
        scales_dir = "src/data/scales"
        
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
    
    def check_note_timing(self, detected_note: str, confidence: float = None, 
                          waveform_data: dict = None, algorithm: str = None,
                          audio_chunk: np.ndarray = None) -> Tuple[bool, float, float]:
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
            
            # Store additional metrics
            if confidence is not None:
                self.confidence_scores.append(confidence)
            else:
                self.confidence_scores.append(0.0)
                
            if waveform_data is not None:
                self.waveform_metrics.append(waveform_data)
            else:
                self.waveform_metrics.append({})
                
            if algorithm is not None:
                self.algorithm_used.append(algorithm)
            else:
                self.algorithm_used.append("unknown")
                
            if audio_chunk is not None:
                self.audio_chunks.append(audio_chunk.copy())
            else:
                self.audio_chunks.append(None)
            
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
        
        # Calculate timing consistency score
        timing_consistency = 0.0
        if len(self.note_delays) > 1:
            # Lower standard deviation = more consistent timing
            delays_without_first = self.note_delays[1:]
            if delays_without_first:
                timing_consistency = 1.0 / (1.0 + np.std(delays_without_first))
        
        # Calculate average confidence
        avg_confidence = 0.0
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        
        return {
            "status": "training",
            "scale_name": self.scale_data["scale_name"],
            "tempo_bpm": self.tempo_bpm,
            "current_note_index": self.current_note_index,
            "total_notes": total_notes,
            "completed_notes": completed_notes,
            "progress_percentage": (completed_notes / total_notes) * 100,
            "average_delay": avg_delay,
            "timing_consistency": timing_consistency,
            "average_confidence": avg_confidence,
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

    def get_comprehensive_analysis_data(self) -> Dict:
        """Get comprehensive data for LLM analysis after training session."""
        if not self.is_training and len(self.actual_timings) == 0:
            return {}
        
        # Calculate detailed statistics
        total_notes = len(self.actual_timings)
        
        # Timing analysis
        timing_stats = {
            "total_notes": total_notes,
            "note_delays": self.note_delays,
            "average_delay": sum(abs(d) for d in self.note_delays[1:]) / max(len(self.note_delays) - 1, 1) if len(self.note_delays) > 1 else 0.0,
            "max_delay": max(self.note_delays[1:]) if len(self.note_delays) > 1 else 0.0,
            "min_delay": min(self.note_delays[1:]) if len(self.note_delays) - 1 > 0 else 0.0,
            "timing_consistency": 1.0 / (1.0 + np.std(self.note_delays[1:])) if len(self.note_delays) > 1 else 0.0
        }
        
        # Audio quality analysis
        audio_stats = {
            "confidence_scores": self.confidence_scores,
            "average_confidence": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            "min_confidence": min(self.confidence_scores) if self.confidence_scores else 0.0,
            "max_confidence": max(self.confidence_scores) if self.confidence_scores else 0.0,
            "algorithm_performance": self.algorithm_used,
            "waveform_metrics": self.waveform_metrics
        }
        
        # Performance summary
        performance_summary = {
            "scale_name": self.scale_data["scale_name"],
            "tempo_bpm": self.tempo_bpm,
            "session_duration": self.actual_timings[-1] if self.actual_timings else 0.0,
            "notes_completed": total_notes,
            "overall_rating": self._calculate_overall_rating(timing_stats, audio_stats)
        }
        
        return {
            "performance_summary": performance_summary,
            "timing_analysis": timing_stats,
            "audio_quality": audio_stats,
            "raw_data": {
                "actual_timings": self.actual_timings,
                "note_timestamps": self.note_timestamps
            }
        }
    
    def _calculate_overall_rating(self, timing_stats: dict, audio_stats: dict) -> str:
        """Calculate an overall performance rating."""
        timing_score = timing_stats["timing_consistency"]
        confidence_score = audio_stats["average_confidence"]
        
        # Combine scores (timing is 60%, confidence is 40%)
        overall_score = (timing_score * 0.6) + (confidence_score * 0.4)
        
        if overall_score >= 0.8:
            return "Excellent"
        elif overall_score >= 0.6:
            return "Good"
        elif overall_score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
