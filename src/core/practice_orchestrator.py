"""Autonomous practice orchestrator for managing complete learning experiences."""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.music_agent import MusicAgent, PracticeFocus, LearningStage, StudentProfile
from src.core.conversation_manager import ConversationManager
from src.core.scale_trainer import ScaleTrainer
from src.core.audio_processor import analyze_waveform_quality
from src.data.config import NOTE_TIMING_TOLERANCE


class PracticePhase(Enum):
    """Different phases of a practice session."""
    WARMUP = "warmup"
    MAIN_PRACTICE = "main_practice"
    COOLDOWN = "cooldown"
    ASSESSMENT = "assessment"
    REFLECTION = "reflection"


class LearningObjective(Enum):
    """Different types of learning objectives."""
    MASTER_SCALE = "master_scale"
    IMPROVE_TIMING = "improve_timing"
    ENHANCE_TECHNIQUE = "enhance_technique"
    BUILD_CONFIDENCE = "build_confidence"
    LEARN_THEORY = "learn_theory"


@dataclass
class PracticeSession:
    """Complete practice session managed by the orchestrator."""
    session_id: str
    start_time: float
    student_profile: StudentProfile
    learning_objectives: List[LearningObjective]
    current_phase: PracticePhase
    phases_completed: List[PracticePhase]
    current_scale: str
    tempo_bpm: int
    focus_areas: List[PracticeFocus]
    performance_metrics: Dict[str, Any]
    session_notes: List[str]
    agent_guidance: List[str]
    
    def __post_init__(self):
        if self.phases_completed is None:
            self.phases_completed = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.session_notes is None:
            self.session_notes = []
        if self.agent_guidance is None:
            self.agent_guidance = []


class PracticeOrchestrator:
    """Autonomous orchestrator for complete music learning experiences."""
    
    def __init__(self, music_agent: MusicAgent, conversation_manager: ConversationManager):
        """Initialize the practice orchestrator."""
        self.music_agent = music_agent
        self.conversation_manager = conversation_manager
        self.scale_trainer = ScaleTrainer()
        
        # Session management
        self.current_session: Optional[PracticeSession] = None
        self.session_history: List[Dict[str, Any]] = []
        
        # Learning progression tracking
        self.learning_paths = self._create_learning_paths()
        self.adaptive_difficulty = True
        
        # Performance thresholds
        self.mastery_threshold = 0.85  # 85% performance required to advance
        self.improvement_threshold = 0.1  # 10% improvement required to continue current path
    
    def _create_learning_paths(self) -> Dict[str, Dict[str, Any]]:
        """Create structured learning paths for different skill levels."""
        return {
            "beginner": {
                "scales": ["C Major Scale", "G Major Scale"],
                "focus_areas": [PracticeFocus.TIMING, PracticeFocus.INTONATION],
                "tempo_range": (30, 80),
                "session_duration": 20,  # minutes
                "phases": [PracticePhase.WARMUP, PracticePhase.MAIN_PRACTICE, PracticePhase.COOLDOWN]
            },
            "intermediate": {
                "scales": ["D Major Scale", "A Minor Scale", "E Minor Scale"],
                "focus_areas": [PracticeFocus.TIMING, PracticeFocus.TECHNIQUE, PracticeFocus.RHYTHM],
                "tempo_range": (60, 120),
                "session_duration": 30,
                "phases": [PracticePhase.WARMUP, PracticePhase.MAIN_PRACTICE, PracticePhase.ASSESSMENT, PracticePhase.COOLDOWN]
            },
            "advanced": {
                "scales": ["All Scales"],
                "focus_areas": [PracticeFocus.TIMING, PracticeFocus.TECHNIQUE, PracticeFocus.RHYTHM, PracticeFocus.EXPRESSION],
                "tempo_range": (80, 180),
                "session_duration": 45,
                "phases": [PracticePhase.WARMUP, PracticePhase.MAIN_PRACTICE, PracticePhase.ASSESSMENT, PracticePhase.REFLECTION, PracticePhase.COOLDOWN]
            }
        }
    
    def start_autonomous_session(self, student_input: str = None) -> PracticeSession:
        """Start a completely autonomous practice session."""
        try:
            # Store student input for parsing
            self._student_input = student_input
            
            # Assess student level if not already done
            if not self.music_agent.student_profile.level:
                self._assess_student_level()
            
            # Determine learning objectives
            objectives = self._determine_learning_objectives(student_input)
            
            # Select appropriate scale and settings
            scale_name, tempo, focus_areas = self._select_session_parameters(objectives)
            
            # Create session
            session_id = f"auto_session_{int(time.time())}"
            self.current_session = PracticeSession(
                session_id=session_id,
                start_time=time.time(),
                student_profile=self.music_agent.student_profile,
                learning_objectives=objectives,
                current_phase=PracticePhase.WARMUP,
                phases_completed=[],
                current_scale=scale_name,
                tempo_bpm=tempo,
                focus_areas=focus_areas,
                performance_metrics={},
                session_notes=[],
                agent_guidance=[]
            )
            
            # Start the scale trainer
            self.scale_trainer.load_scale_by_name(scale_name)
            self.scale_trainer.set_tempo(tempo)
            self.scale_trainer.start_training()
            
            # Generate initial guidance
            initial_guidance = self._generate_phase_guidance(PracticePhase.WARMUP)
            self.current_session.agent_guidance.append(initial_guidance)
            
            return self.current_session
            
        except Exception as e:
            print(f"Error starting autonomous session: {e}")
            return None
    
    def _assess_student_level(self):
        """Assess the student's current skill level."""
        # Use default assessment or previous performance data
        if hasattr(self.music_agent, 'learning_history') and self.music_agent.learning_history:
            # Analyze recent performance
            recent_sessions = [h for h in self.music_agent.learning_history if h.get('type') == 'practice_session']
            if recent_sessions:
                latest_session = recent_sessions[-1]
                performance_data = latest_session.get('data', {}).get('performance_data', {})
                
                # Assess level based on performance
                if performance_data:
                    self.music_agent.get_agent_guidance(
                        "Please assess my current skill level based on my recent performance",
                        {"performance_data": json.dumps(performance_data)}
                    )
        else:
            # Default to beginner if no history
            self.music_agent.student_profile.level = LearningStage.BEGINNER
    
    def _determine_learning_objectives(self, student_input: str = None) -> List[LearningObjective]:
        """Determine what the student should focus on learning."""
        objectives = []
        
        # Base objectives based on level
        level = self.music_agent.student_profile.level
        if level == LearningStage.BEGINNER:
            objectives.extend([LearningObjective.MASTER_SCALE, LearningObjective.BUILD_CONFIDENCE])
        elif level == LearningStage.INTERMEDIATE:
            objectives.extend([LearningObjective.IMPROVE_TIMING, LearningObjective.ENHANCE_TECHNIQUE])
        else:  # Advanced
            objectives.extend([LearningObjective.ENHANCE_TECHNIQUE, LearningObjective.LEARN_THEORY])
        
        # Add objectives based on student input
        if student_input:
            if "timing" in student_input.lower():
                objectives.append(LearningObjective.IMPROVE_TIMING)
            if "technique" in student_input.lower():
                objectives.append(LearningObjective.ENHANCE_TECHNIQUE)
            if "confidence" in student_input.lower():
                objectives.append(LearningObjective.BUILD_CONFIDENCE)
        
        return list(set(objectives))  # Remove duplicates
    
    def _select_session_parameters(self, objectives: List[LearningObjective]) -> Tuple[str, int, List[PracticeFocus]]:
        """Select appropriate scale, tempo, and focus areas for the session."""
        level = self.music_agent.student_profile.level
        level_config = self.learning_paths.get(level.value, self.learning_paths["beginner"])
        
        # Parse user input for specific scale and tempo preferences
        user_scale = None
        user_tempo = None
        
        # Check if we have student input to parse
        if hasattr(self, '_student_input') and self._student_input:
            input_lower = self._student_input.lower()
            
            # Extract scale preference
            scale_patterns = {
                "c major": "C Major Scale",
                "g major": "G Major Scale", 
                "d major": "D Major Scale",
                "a minor": "A Minor Scale",
                "e minor": "E Minor Scale",
                "c major scale": "C Major Scale",
                "g major scale": "G Major Scale",
                "d major scale": "D Major Scale",
                "a minor scale": "A Minor Scale",
                "e minor scale": "E Minor Scale"
            }
            
            for pattern, scale_name in scale_patterns.items():
                if pattern in input_lower:
                    user_scale = scale_name
                    break
            
            # Extract tempo preference
            import re
            tempo_match = re.search(r'(\d+)\s*bpm', input_lower)
            if tempo_match:
                user_tempo = int(tempo_match.group(1))
            else:
                # Look for other tempo indicators
                if "slow" in input_lower or "slowly" in input_lower:
                    user_tempo = 60
                elif "fast" in input_lower or "quickly" in input_lower:
                    user_tempo = 120
                elif "medium" in input_lower or "moderate" in input_lower:
                    user_tempo = 90
        
        # Select scale
        if user_scale and user_scale in [scale['name'] for scale in self.scale_trainer.get_available_scales()]:
            # Use user's preferred scale if available
            scale_name = user_scale
        else:
            # Fall back to AI selection logic
            available_scales = level_config["scales"]
            if "All Scales" in available_scales:
                # Let the agent recommend the best scale
                current_scale = self.scale_trainer.scale_data.get("scale_name", "C Major Scale")
                recommendation = self.music_agent.get_agent_guidance(
                    f"Recommend the next scale I should practice after {current_scale}",
                    {"current_scale": current_scale, "performance_rating": "good"}
                )
                # Parse recommendation to get scale name
                scale_name = self._extract_scale_from_recommendation(recommendation) or available_scales[0]
            else:
                # Select based on completion status
                completed = self.music_agent.student_profile.completed_scales
                for scale in available_scales:
                    if scale not in completed:
                        scale_name = scale
                        break
                else:
                    scale_name = available_scales[0]
        
        # Select tempo
        if user_tempo:
            # Use user's preferred tempo, but ensure it's within level-appropriate range
            min_tempo, max_tempo = level_config["tempo_range"]
            tempo = max(min_tempo, min(max_tempo, user_tempo))
        else:
            # AI selects tempo based on objectives
            min_tempo, max_tempo = level_config["tempo_range"]
            if LearningObjective.IMPROVE_TIMING in objectives:
                # Lower tempo for timing focus
                tempo = max(min_tempo, (min_tempo + max_tempo) // 3)
            else:
                # Moderate tempo for general practice
                tempo = (min_tempo + max_tempo) // 2
        
        # Select focus areas
        focus_areas = level_config["focus_areas"].copy()
        
        # Adjust based on objectives
        if LearningObjective.IMPROVE_TIMING in objectives:
            if PracticeFocus.TIMING not in focus_areas:
                focus_areas.append(PracticeFocus.TIMING)
        if LearningObjective.ENHANCE_TECHNIQUE in objectives:
            if PracticeFocus.TECHNIQUE not in focus_areas:
                focus_areas.append(PracticeFocus.TECHNIQUE)
        
        return scale_name, tempo, focus_areas
    
    def _extract_scale_from_recommendation(self, recommendation: str) -> Optional[str]:
        """Extract scale name from agent recommendation."""
        # Simple extraction - look for scale patterns
        scale_patterns = [
            "C Major Scale", "G Major Scale", "D Major Scale", "A Minor Scale", "E Minor Scale"
        ]
        
        for pattern in scale_patterns:
            if pattern.lower() in recommendation.lower():
                return pattern
        
        return None
    
    def _generate_phase_guidance(self, phase: PracticePhase) -> str:
        """Generate guidance for a specific practice phase."""
        phase_guidance = {
            PracticePhase.WARMUP: "Let's start with a gentle warmup. Play the scale slowly and focus on clean, clear notes.",
            PracticePhase.MAIN_PRACTICE: "Now let's work on the main practice. Focus on your target areas and maintain steady tempo.",
            PracticePhase.ASSESSMENT: "Let's assess your progress. Play the scale at your target tempo and I'll analyze your performance.",
            PracticePhase.REFLECTION: "Take a moment to reflect on what you've learned and how you can apply it.",
            PracticePhase.COOLDOWN: "Great work! Let's finish with a relaxed cooldown. Play the scale slowly and enjoy the music."
        }
        
        return phase_guidance.get(phase, "Continue with your practice.")
    
    def advance_phase(self) -> bool:
        """Advance to the next practice phase."""
        if not self.current_session:
            return False
        
        # Mark current phase as completed
        self.current_session.phases_completed.append(self.current_session.current_phase)
        
        # Determine next phase
        level = self.music_agent.student_profile.level
        level_config = self.learning_paths.get(level.value, self.learning_paths["beginner"])
        available_phases = level_config["phases"]
        
        current_index = available_phases.index(self.current_session.current_phase)
        if current_index + 1 < len(available_phases):
            next_phase = available_phases[current_index + 1]
            self.current_session.current_phase = next_phase
            
            # Generate guidance for new phase
            guidance = self._generate_phase_guidance(next_phase)
            self.current_session.agent_guidance.append(guidance)
            
            return True
        else:
            # Session complete
            return False
    
    def update_performance_metrics(self, performance_data: Dict[str, Any]):
        """Update performance metrics during the session."""
        if not self.current_session:
            return
        
        # Update current metrics
        self.current_session.performance_metrics.update(performance_data)
        
        # Check if ready to advance phase
        if self._should_advance_phase(performance_data):
            self.advance_phase()
    
    def _should_advance_phase(self, performance_data: Dict[str, Any]) -> bool:
        """Determine if the student is ready to advance to the next phase."""
        current_phase = self.current_session.current_phase
        
        if current_phase == PracticePhase.WARMUP:
            # Advance if basic notes are being played
            detected_notes = performance_data.get("detected_notes", [])
            return len(detected_notes) >= 3  # At least 3 notes detected
        
        elif current_phase == PracticePhase.MAIN_PRACTICE:
            # Advance if performance meets threshold
            timing_consistency = performance_data.get("timing_consistency", 0.0)
            avg_confidence = performance_data.get("average_confidence", 0.0)
            performance_score = (timing_consistency * 0.6) + (avg_confidence * 0.4)
            return performance_score >= self.mastery_threshold
        
        elif current_phase == PracticePhase.ASSESSMENT:
            # Advance after assessment is complete
            return True
        
        elif current_phase == PracticePhase.REFLECTION:
            # Advance after reflection period
            return True
        
        return False
    
    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        return self.current_session is not None
    
    def get_current_guidance(self) -> str:
        """Get the current guidance for the student."""
        if not self.current_session or not self.current_session.agent_guidance:
            return "Ready to start your practice session!"
        
        return self.current_session.agent_guidance[-1]
    
    def add_session_note(self, note: str):
        """Add a note to the current session."""
        if self.current_session:
            self.current_session.session_notes.append(note)
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and generate summary."""
        if not self.current_session:
            return {"error": "No active session to end"}
        
        try:
            # Calculate session duration
            session_duration = time.time() - self.current_session.start_time
            
            # Generate final assessment
            final_assessment = self._generate_final_assessment()
            
            # Create session summary
            session_summary = {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "end_time": time.time(),
                "duration_minutes": session_duration / 60,
                "learning_objectives": [obj.value for obj in self.current_session.learning_objectives],
                "phases_completed": [phase.value for phase in self.current_session.phases_completed],
                "current_scale": self.current_session.current_scale,
                "tempo_bpm": self.current_session.tempo_bpm,
                "focus_areas": [area.value for area in self.current_session.focus_areas],
                "performance_metrics": self.current_session.performance_metrics,
                "session_notes": self.current_session.session_notes,
                "agent_guidance": self.current_session.agent_guidance,
                "final_assessment": final_assessment
            }
            
            # Add to history
            self.session_history.append(session_summary)
            
            # Update student profile
            if self.current_session.current_scale not in self.music_agent.student_profile.completed_scales:
                self.music_agent.student_profile.completed_scales.append(self.current_session.current_scale)
            
            # Clear current session
            self.current_session = None
            
            # Stop scale training
            self.scale_trainer.stop_training()
            
            return session_summary
            
        except Exception as e:
            # If there's an error, still clear the session
            self.current_session = None
            return {"error": f"Error ending session: {str(e)}"}
    
    def _generate_final_assessment(self) -> str:
        """Generate a final assessment of the session."""
        if not self.current_session:
            return "No session data available for assessment."
        
        try:
            # Get performance data
            performance_data = self.current_session.performance_metrics
            objectives = self.current_session.learning_objectives
            
            # Create assessment prompt
            assessment_prompt = f"""Assess this practice session:

Objectives: {[obj.value for obj in objectives]}
Scale: {self.current_session.current_scale}
Tempo: {self.current_session.tempo_bpm} BPM
Focus Areas: {[area.value for area in self.current_session.focus_areas]}
Performance Data: {json.dumps(performance_data, indent=2)}

Provide a brief, encouraging assessment of what was accomplished and suggestions for next steps."""

            # Get agent assessment
            assessment = self.music_agent.get_agent_guidance(assessment_prompt)
            return assessment
            
        except Exception as e:
            return f"Session completed successfully! Great work on {self.current_session.current_scale}."
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """Get current session information for UI display."""
        if not self.current_session:
            return {"status": "no_session"}
        
        return {
            "status": "active",
            "session_id": self.current_session.session_id,
            "current_phase": self.current_session.current_phase.value,
            "phases_completed": [phase.value for phase in self.current_session.phases_completed],
            "current_scale": self.current_session.current_scale,
            "tempo_bpm": self.current_session.tempo_bpm,
            "focus_areas": [area.value for area in self.current_session.focus_areas],
            "current_guidance": self.get_current_guidance(),
            "session_duration_minutes": (time.time() - self.current_session.start_time) / 60,
            "learning_objectives": [obj.value for obj in self.current_session.learning_objectives]
        }
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        if not self.current_session:
            return {"status": "no_session"}
        
        return {
            "status": "active",
            "session_id": self.current_session.session_id,
            "current_phase": self.current_session.current_phase.value,
            "phases_completed": [phase.value for phase in self.current_session.phases_completed],
            "current_scale": self.current_session.current_scale,
            "tempo_bpm": self.current_session.tempo_bpm,
            "focus_areas": [area.value for area in self.current_session.focus_areas],
            "current_guidance": self.get_current_guidance(),
            "session_duration_minutes": (time.time() - self.current_session.start_time) / 60
        }
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get comprehensive learning progress overview."""
        return {
            "student_profile": self.music_agent.get_learning_progress()["student_profile"],
            "session_history": self.session_history,
            "current_session": self.get_session_status(),
            "learning_paths": self.learning_paths
        }


def create_practice_orchestrator(music_agent: MusicAgent, conversation_manager: ConversationManager) -> PracticeOrchestrator:
    """Factory function to create a practice orchestrator."""
    try:
        orchestrator = PracticeOrchestrator(music_agent, conversation_manager)
        return orchestrator
    except Exception as e:
        print(f"Error creating practice orchestrator: {e}")
        return None
