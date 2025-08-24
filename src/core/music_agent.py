"""LangChain-powered Music Agent for autonomous music tutoring."""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent import AgentAction
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.core.scale_trainer import ScaleTrainer
from src.core.audio_processor import analyze_waveform_quality
from src.data.config import NOTE_TIMING_TOLERANCE


class LearningStage(Enum):
    """Enum for different learning stages."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PracticeFocus(Enum):
    """Enum for different practice focus areas."""
    TIMING = "timing"
    INTONATION = "intonation"
    TECHNIQUE = "technique"
    RHYTHM = "rhythm"
    EXPRESSION = "expression"


@dataclass
class StudentProfile:
    """Student profile and learning history."""
    name: str = "Student"
    level: LearningStage = LearningStage.BEGINNER
    preferred_instrument: str = "guitar"
    practice_time_per_day: int = 30  # minutes
    strengths: List[str] = None
    weaknesses: List[str] = None
    completed_scales: List[str] = None
    current_goals: List[str] = None
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
        if self.completed_scales is None:
            self.completed_scales = []
        if self.current_goals is None:
            self.current_goals = []


@dataclass
class PracticeSession:
    """Practice session data and goals."""
    session_id: str
    start_time: float
    scale_name: str
    tempo_bpm: int
    focus_areas: List[PracticeFocus]
    target_improvements: List[str]
    session_notes: List[str] = None
    
    def __post_init__(self):
        if self.session_notes is None:
            self.session_notes = []


class MusicAgent:
    """LangChain-powered autonomous music tutor agent."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b"):
        """Initialize the music agent."""
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, base_url=ollama_url)
        
        # Student and session management
        self.student_profile = StudentProfile()
        self.current_session: Optional[PracticeSession] = None
        self.learning_history: List[Dict] = []
        
        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        # Simplified approach - no agent executor for now
        self.agent_executor = None
        
        # Learning progression tracking
        self.scale_progression = {
            "C Major Scale": {"difficulty": 1, "prerequisites": [], "next_scales": ["G Major Scale", "F Major Scale"]},
            "G Major Scale": {"difficulty": 2, "prerequisites": ["C Major Scale"], "next_scales": ["D Major Scale", "A Major Scale"]},
            "D Major Scale": {"difficulty": 3, "prerequisites": ["G Major Scale"], "next_scales": ["A Major Scale", "E Major Scale"]},
            "A Minor Scale": {"difficulty": 2, "prerequisites": ["C Major Scale"], "next_scales": ["E Minor Scale", "D Minor Scale"]},
            "E Minor Scale": {"difficulty": 3, "prerequisites": ["A Minor Scale"], "next_scales": ["B Minor Scale", "F# Minor Scale"]}
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools for the music agent."""
        
        @tool
        def assess_student_level(performance_data: str) -> str:
            """Assess student's current skill level based on performance data."""
            try:
                data = json.loads(performance_data)
                timing_consistency = data.get("timing_analysis", {}).get("timing_consistency", 0.0)
                avg_confidence = data.get("audio_quality", {}).get("average_confidence", 0.0)
                
                # Calculate overall skill score
                skill_score = (timing_consistency * 0.6) + (avg_confidence * 0.4)
                
                if skill_score >= 0.8:
                    level = LearningStage.ADVANCED
                elif skill_score >= 0.6:
                    level = LearningStage.INTERMEDIATE
                else:
                    level = LearningStage.BEGINNER
                
                self.student_profile.level = level
                return f"Student assessed as {level.value} level (score: {skill_score:.2f})"
                
            except Exception as e:
                return f"Error assessing level: {str(e)}"
        
        @tool
        def recommend_next_scale(current_scale: str, performance_rating: str) -> str:
            """Recommend the next scale to practice based on current performance."""
            try:
                if current_scale not in self.scale_progression:
                    return "Unknown scale. Available scales: " + ", ".join(self.scale_progression.keys())
                
                current_info = self.scale_progression[current_scale]
                available_next = current_info["next_scales"]
                
                if not available_next:
                    return "Congratulations! You've completed all available scales."
                
                # Filter based on prerequisites
                eligible_scales = []
                for scale in available_next:
                    prereqs = self.scale_progression[scale]["prerequisites"]
                    if all(prereq in self.student_profile.completed_scales for prereq in prereqs):
                        eligible_scales.append(scale)
                
                if not eligible_scales:
                    return f"Complete more prerequisites first. Focus on: {', '.join(current_info['prerequisites'])}"
                
                # Recommend based on difficulty progression
                recommended = min(eligible_scales, key=lambda x: self.scale_progression[x]["difficulty"])
                return f"Recommended next scale: {recommended} (difficulty: {self.scale_progression[recommended]['difficulty']})"
                
            except Exception as e:
                return f"Error recommending scale: {str(e)}"
        
        @tool
        def create_practice_plan(focus_areas: str, duration_minutes: int) -> str:
            """Create a personalized practice plan for the student."""
            try:
                focus_list = [PracticeFocus(focus.strip()) for focus in focus_areas.split(",")]
                
                plan = {
                    "warmup": "5 minutes - Simple finger exercises and basic scales",
                    "main_practice": f"{duration_minutes - 10} minutes - Focus on: {', '.join([f.value for f in focus_list])}",
                    "cooldown": "5 minutes - Slow, relaxed playing and review"
                }
                
                # Add specific exercises based on focus areas
                specific_exercises = []
                for focus in focus_list:
                    if focus == PracticeFocus.TIMING:
                        specific_exercises.append("Metronome practice with quarter notes")
                    elif focus == PracticeFocus.INTONATION:
                        specific_exercises.append("Long tone exercises with tuner")
                    elif focus == PracticeFocus.TECHNIQUE:
                        specific_exercises.append("Finger independence drills")
                    elif focus == PracticeFocus.RHYTHM:
                        specific_exercises.append("Rhythm pattern practice")
                    elif focus == PracticeFocus.EXPRESSION:
                        specific_exercises.append("Dynamic control exercises")
                
                plan["specific_exercises"] = specific_exercises
                
                return json.dumps(plan, indent=2)
                
            except Exception as e:
                return f"Error creating practice plan: {str(e)}"
        
        @tool
        def analyze_practice_patterns(session_history: str) -> str:
            """Analyze practice patterns and suggest improvements."""
            try:
                history = json.loads(session_history)
                
                # Analyze common patterns
                total_sessions = len(history)
                avg_duration = sum(s.get("duration", 0) for s in history) / max(total_sessions, 1)
                common_focus = {}
                
                for session in history:
                    for focus in session.get("focus_areas", []):
                        common_focus[focus] = common_focus.get(focus, 0) + 1
                
                # Identify patterns
                most_common_focus = max(common_focus.items(), key=lambda x: x[1]) if common_focus else ("none", 0)
                least_common_focus = min(common_focus.items(), key=lambda x: x[1]) if common_focus else ("none", 0)
                
                analysis = {
                    "total_sessions": total_sessions,
                    "average_duration_minutes": avg_duration,
                    "most_practiced_area": most_common_focus[0],
                    "least_practiced_area": least_common_focus[0],
                    "recommendations": []
                }
                
                # Generate recommendations
                if avg_duration < 20:
                    analysis["recommendations"].append("Consider longer practice sessions for better skill development")
                
                if most_common_focus[1] > total_sessions * 0.7:
                    analysis["recommendations"].append(f"Balance your practice - you focus heavily on {most_common_focus[0]}")
                
                if least_common_focus[1] < total_sessions * 0.2:
                    analysis["recommendations"].append(f"Consider more practice in {least_common_focus[0]} area")
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                return f"Error analyzing patterns: {str(e)}"
        
        @tool
        def set_practice_goals(goals: str, timeframe_days: int) -> str:
            """Set specific practice goals for the student."""
            try:
                goal_list = [goal.strip() for goal in goals.split(",")]
                
                # Validate goals
                valid_goals = []
                for goal in goal_list:
                    if len(goal) > 5 and len(goal) < 100:  # Reasonable goal length
                        valid_goals.append(goal)
                
                # Update student profile
                self.student_profile.current_goals = valid_goals
                
                # Create goal tracking structure
                goal_tracker = {
                    "goals": valid_goals,
                    "timeframe_days": timeframe_days,
                    "start_date": time.time(),
                    "deadline": time.time() + (timeframe_days * 24 * 3600),
                    "progress": [0.0] * len(valid_goals),  # 0.0 to 1.0 progress
                    "milestones": []
                }
                
                # Add to learning history
                self.learning_history.append({
                    "type": "goal_setting",
                    "timestamp": time.time(),
                    "data": goal_tracker
                })
                
                return f"Set {len(valid_goals)} goals for {timeframe_days} days: {', '.join(valid_goals)}"
                
            except Exception as e:
                return f"Error setting goals: {str(e)}"
        
        return [assess_student_level, recommend_next_scale, create_practice_plan, 
                analyze_practice_patterns, set_practice_goals]
    
    def _create_agent(self):
        """Create the LangChain agent with music tutoring capabilities."""
        
        system_prompt = """You are an expert music teacher and autonomous tutor. Your role is to:

1. **Assess Student Level**: Analyze performance data to determine skill level
2. **Guide Learning Progression**: Recommend appropriate scales and exercises
3. **Create Practice Plans**: Design personalized practice sessions
4. **Set Goals**: Establish clear, achievable learning objectives
5. **Provide Motivation**: Encourage and support student progress

You have access to tools for:
- Assessing student skill level from performance data
- Recommending next scales based on current progress
- Creating personalized practice plans
- Analyzing practice patterns
- Setting and tracking learning goals

Always be encouraging, specific, and actionable in your advice. Use musical terminology appropriately and consider the student's current level when making recommendations."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use a simpler approach for now - direct LLM calls instead of agent executor
        return None  # We'll handle this differently
    
    def start_practice_session(self, scale_name: str, tempo_bpm: int, focus_areas: List[PracticeFocus]) -> PracticeSession:
        """Start a new practice session with the agent's guidance."""
        session_id = f"session_{int(time.time())}"
        
        self.current_session = PracticeSession(
            session_id=session_id,
            start_time=time.time(),
            scale_name=scale_name,
            tempo_bpm=tempo_bpm,
            focus_areas=focus_areas,
            target_improvements=[]
        )
        
        # Generate session-specific goals
        session_goals = self._generate_session_goals(scale_name, focus_areas)
        self.current_session.target_improvements = session_goals
        
        return self.current_session
    
    def _generate_session_goals(self, scale_name: str, focus_areas: List[PracticeFocus]) -> List[str]:
        """Generate specific goals for the current practice session."""
        goals = []
        
        for focus in focus_areas:
            if focus == PracticeFocus.TIMING:
                goals.append(f"Maintain consistent timing within {NOTE_TIMING_TOLERANCE}s tolerance")
            elif focus == PracticeFocus.INTONATION:
                goals.append("Play each note with clear, accurate pitch")
            elif focus == PracticeFocus.TECHNIQUE:
                goals.append("Use proper finger placement and hand position")
            elif focus == PracticeFocus.RHYTHM:
                goals.append("Keep steady rhythm matching the metronome")
            elif focus == PracticeFocus.EXPRESSION:
                goals.append("Add musical expression and dynamics")
        
        # Add scale-specific goals
        if "Major" in scale_name:
            goals.append("Emphasize the major scale's bright, happy character")
        elif "Minor" in scale_name:
            goals.append("Express the minor scale's more somber, introspective quality")
        
        return goals
    
    def get_agent_guidance(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Get autonomous guidance from the music agent."""
        try:
            # Prepare input with context
            full_input = user_input
            if context:
                context_str = json.dumps(context, indent=2)
                full_input += f"\n\nContext:\n{context_str}"
            
            # Use direct LLM call instead of agent executor
            system_prompt = """You are an expert music teacher. Provide helpful, encouraging advice for music students. Be specific and actionable in your guidance."""
            
            # Simple prompt template
            prompt = f"{system_prompt}\n\nStudent Question: {full_input}\n\nYour Response:"
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            return str(response).strip()
            
        except Exception as e:
            return f"I'm here to help with your music practice! What would you like to work on today?"
    
    def end_practice_session(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """End the current practice session and generate insights."""
        if not self.current_session:
            return {"error": "No active session to end"}
        
        # Calculate session duration
        session_duration = time.time() - self.current_session.start_time
        
        # Analyze performance
        session_summary = {
            "session_id": self.current_session.session_id,
            "scale_name": self.current_session.scale_name,
            "tempo_bpm": self.current_session.tempo_bpm,
            "duration_minutes": session_duration / 60,
            "focus_areas": [f.value for f in self.current_session.focus_areas],
            "target_improvements": self.current_session.target_improvements,
            "performance_data": performance_data,
            "session_notes": self.current_session.session_notes
        }
        
        # Add to learning history
        self.learning_history.append({
            "type": "practice_session",
            "timestamp": time.time(),
            "data": session_summary
        })
        
        # Update student profile
        if self.current_session.scale_name not in self.student_profile.completed_scales:
            self.student_profile.completed_scales.append(self.current_session.scale_name)
        
        # Clear current session
        self.current_session = None
        
        return session_summary
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get comprehensive learning progress overview."""
        return {
            "student_profile": {
                "name": self.student_profile.name,
                "level": self.student_profile.level.value,
                "preferred_instrument": self.student_profile.preferred_instrument,
                "practice_time_per_day": self.student_profile.practice_time_per_day,
                "strengths": self.student_profile.strengths,
                "weaknesses": self.student_profile.weaknesses,
                "completed_scales": self.student_profile.completed_scales,
                "current_goals": self.student_profile.current_goals
            },
            "learning_history": self.learning_history,
            "scale_progression": self.scale_progression
        }


def create_music_agent(ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b") -> MusicAgent:
    """Factory function to create a music agent with error handling."""
    try:
        agent = MusicAgent(ollama_url, model_name)
        return agent
    except Exception as e:
        print(f"Error creating music agent: {e}")
        return None
