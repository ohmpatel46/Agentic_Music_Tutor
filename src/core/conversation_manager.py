"""Conversation manager for natural language interaction with the music agent."""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from src.core.music_agent import MusicAgent, PracticeFocus, LearningStage


class ConversationType(Enum):
    """Types of conversations the agent can handle."""
    GREETING = "greeting"
    PRACTICE_GUIDANCE = "practice_guidance"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    GOAL_SETTING = "goal_setting"
    PROGRESS_REVIEW = "progress_review"
    MOTIVATION = "motivation"
    TECHNICAL_HELP = "technical_help"
    GENERAL_QUESTION = "general_question"


@dataclass
class ConversationContext:
    """Context for ongoing conversations."""
    conversation_id: str
    conversation_type: ConversationType
    start_time: float
    messages: List[Dict[str, Any]]
    current_topic: str
    student_mood: str = "neutral"
    practice_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.practice_context is None:
            self.practice_context = {}


class ConversationManager:
    """Manages natural language conversations with the music agent."""
    
    def __init__(self, music_agent: MusicAgent, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b"):
        """Initialize the conversation manager."""
        self.music_agent = music_agent
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, base_url=ollama_url)
        
        # Conversation tracking
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Conversation templates
        self.conversation_templates = self._create_conversation_templates()
        
        # Intent recognition patterns
        self.intent_patterns = self._create_intent_patterns()
    
    def _create_conversation_templates(self) -> Dict[str, str]:
        """Create conversation templates for different types of interactions."""
        return {
            "greeting": """You are a friendly, encouraging music teacher. Greet the student warmly and ask how you can help them today with their music practice.""",
            
            "practice_guidance": """You are an expert music teacher helping a student with their practice. Provide specific, actionable advice based on their current practice session and goals. Use musical terminology appropriately and be encouraging.""",
            
            "performance_analysis": """You are analyzing a student's music performance. Provide detailed, constructive feedback on their timing, technique, and overall performance. Be specific about what went well and what can be improved.""",
            
            "goal_setting": """You are helping a student set music practice goals. Help them create realistic, achievable goals that will help them progress. Consider their current skill level and available practice time.""",
            
            "progress_review": """You are reviewing a student's learning progress. Highlight their achievements, identify areas for improvement, and suggest next steps in their musical journey.""",
            
            "motivation": """You are motivating a student who may be feeling discouraged or stuck. Provide encouragement, remind them of their progress, and help them find renewed enthusiasm for their music practice.""",
            
            "technical_help": """You are providing technical assistance with music theory, technique, or instrument-specific questions. Give clear, practical explanations that the student can apply immediately.""",
            
            "general_question": """You are answering general questions about music, practice, or learning. Provide helpful, accurate information in an encouraging tone."""
        }
    
    def _create_intent_patterns(self) -> Dict[str, List[str]]:
        """Create patterns for recognizing user intent."""
        return {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "start", "begin"],
            "practice_guidance": ["help", "practice", "what should i", "how do i", "advice", "suggestion", "tip"],
            "performance_analysis": ["how did i do", "analyze", "feedback", "review", "assessment", "evaluation"],
            "goal_setting": ["goal", "target", "aim", "objective", "plan", "schedule", "timeline"],
            "progress_review": ["progress", "improvement", "how am i doing", "status", "update", "check"],
            "motivation": ["discouraged", "stuck", "frustrated", "tired", "bored", "motivate", "encourage"],
            "technical_help": ["theory", "technique", "how to", "explain", "what is", "why does", "technical"],
            "general_question": ["question", "what", "when", "where", "why", "how", "tell me about"]
        }
    
    def start_conversation(self, user_input: str, conversation_type: Optional[ConversationType] = None) -> str:
        """Start a new conversation or continue an existing one."""
        try:
            # Detect conversation type if not provided
            if not conversation_type:
                conversation_type = self._detect_conversation_type(user_input)
            
            # Create or get conversation context
            conversation_id = f"conv_{int(time.time())}"
            if conversation_id not in self.active_conversations:
                context = ConversationContext(
                    conversation_id=conversation_id,
                    conversation_type=conversation_type,
                    start_time=time.time(),
                    messages=[],
                    current_topic=user_input[:100]  # First 100 chars as topic
                )
                self.active_conversations[conversation_id] = context
            else:
                context = self.active_conversations[conversation_id]
            
            # Add user message to context
            context.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate response using appropriate template
            response = self._generate_response(user_input, context)
            
            # Add AI response to context
            context.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": time.time()
            })
            
            return response
            
        except Exception as e:
            return f"I'm having trouble with our conversation. Let me try to help: {str(e)}"
    
    def _detect_conversation_type(self, user_input: str) -> ConversationType:
        """Detect the type of conversation from user input."""
        user_input_lower = user_input.lower()
        
        # Score each intent pattern
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in user_input_lower)
            intent_scores[intent] = score
        
        # Get the highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return ConversationType(best_intent[0])
        
        # Default to general question
        return ConversationType.GENERAL_QUESTION
    
    def _generate_response(self, user_input: str, context: ConversationContext) -> str:
        """Generate a response based on conversation context and type."""
        try:
            # Get the appropriate template
            template = self.conversation_templates.get(context.conversation_type.value, 
                                                    self.conversation_templates["general_question"])
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Prepare chat history
            chat_history = []
            for msg in context.messages[:-1]:  # Exclude the current user message
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Generate response
            chain = prompt | self.llm | str
            
            response = chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })
            
            # If this is a practice-related conversation, try to get agent guidance
            if context.conversation_type in [ConversationType.PRACTICE_GUIDANCE, 
                                           ConversationType.PERFORMANCE_ANALYSIS]:
                try:
                    agent_guidance = self.music_agent.get_agent_guidance(user_input, {
                        "conversation_type": context.conversation_type.value,
                        "current_topic": context.current_topic,
                        "student_mood": context.student_mood
                    })
                    
                    # Combine conversational response with agent guidance
                    if agent_guidance and len(agent_guidance) > 20:
                        response += f"\n\n🤖 **AI Tutor Insight:**\n{agent_guidance}"
                        
                except Exception as e:
                    # If agent guidance fails, continue with just the conversational response
                    pass
            
            return response.strip()
            
        except Exception as e:
            return f"I'm here to help with your music practice! What would you like to work on today?"
    
    def continue_conversation(self, conversation_id: str, user_input: str) -> str:
        """Continue an existing conversation."""
        if conversation_id not in self.active_conversations:
            return self.start_conversation(user_input)
        
        context = self.active_conversations[conversation_id]
        
        # Add user message
        context.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Generate response
        response = self._generate_response(user_input, context)
        
        # Add AI response
        context.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        return response
    
    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End a conversation and return summary."""
        if conversation_id not in self.active_conversations:
            return {"error": "Conversation not found"}
        
        context = self.active_conversations[conversation_id]
        
        # Create conversation summary
        summary = {
            "conversation_id": conversation_id,
            "conversation_type": context.conversation_type.value,
            "start_time": context.start_time,
            "end_time": time.time(),
            "duration_minutes": (time.time() - context.start_time) / 60,
            "total_messages": len(context.messages),
            "current_topic": context.current_topic,
            "student_mood": context.student_mood
        }
        
        # Add to history
        self.conversation_history.append(summary)
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        
        return summary
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific conversation."""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            return {
                "conversation_id": conversation_id,
                "conversation_type": context.conversation_type.value,
                "start_time": context.start_time,
                "duration_minutes": (time.time() - context.start_time) / 60,
                "total_messages": len(context.messages),
                "current_topic": context.current_topic,
                "student_mood": context.student_mood
            }
        return None
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all active and historical conversations."""
        active = [self.get_conversation_summary(conv_id) for conv_id in self.active_conversations.keys()]
        active = [conv for conv in active if conv is not None]
        
        return active + self.conversation_history
    
    def update_student_mood(self, conversation_id: str, mood: str):
        """Update the student's mood in a conversation context."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id].student_mood = mood
    
    def add_practice_context(self, conversation_id: str, practice_data: Dict[str, Any]):
        """Add practice-related context to a conversation."""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            context.practice_context.update(practice_data)


def create_conversation_manager(music_agent: MusicAgent, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b") -> ConversationManager:
    """Factory function to create a conversation manager."""
    try:
        manager = ConversationManager(music_agent, ollama_url, model_name)
        return manager
    except Exception as e:
        print(f"Error creating conversation manager: {e}")
        return None
