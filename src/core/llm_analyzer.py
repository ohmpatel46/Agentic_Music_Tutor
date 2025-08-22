"""LLM-powered analysis module for the Agentic Music Tutor."""

import requests
import json
from typing import Dict, Optional, List
import streamlit as st


class LLMAnalyzer:
    """Handles LLM analysis of music performance using Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize the LLM analyzer with Ollama connection."""
        self.ollama_url = ollama_url
        self.model_name = "llama3.2:3b"  # Default model, can be changed
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return ["llama3.2:3b"]  # Fallback
        except Exception as e:
            print(f"Error getting Ollama models: {e}")
            return ["llama3.2:3b"]  # Fallback
    
    def set_model(self, model_name: str):
        """Set the Ollama model to use for analysis."""
        if model_name in self.available_models:
            self.model_name = model_name
            return True
        return False
    
    def analyze_performance(self, analysis_data: Dict) -> Optional[str]:
        """Analyze music performance data and return LLM insights."""
        if not analysis_data:
            return "No performance data available for analysis."
        
        try:
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(analysis_data)
            
            # Send to Ollama
            response = self._query_ollama(prompt)
            
            if response:
                return response
            else:
                return "Unable to get analysis from LLM. Please check Ollama connection."
                
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return f"Error during analysis: {str(e)}"
    
    def _create_analysis_prompt(self, data: Dict) -> str:
        """Create a comprehensive prompt for the LLM analysis."""
        
        # Extract key information
        perf_summary = data.get("performance_summary", {})
        timing_stats = data.get("timing_analysis", {})
        audio_stats = data.get("audio_quality", {})
        
        # Build the prompt
        prompt = f"""You are an expert music teacher analyzing a student's scale performance. 
Provide concise, actionable feedback in exactly 3 sections with bullet points.

PERFORMANCE DATA:
Scale: {perf_summary.get('scale_name', 'Unknown')}
Tempo: {perf_summary.get('tempo_bpm', 0)} BPM
Session Duration: {perf_summary.get('session_duration', 0):.1f}s
Notes Completed: {perf_summary.get('notes_completed', 0)}
Overall Rating: {perf_summary.get('overall_rating', 'Unknown')}

TIMING METRICS:
- Average Delay: {timing_stats.get('average_delay', 0):.2f}s
- Max Delay: {timing_stats.get('max_delay', 0):.2f}s
- Min Delay: {timing_stats.get('min_delay', 0):.2f}s
- Timing Consistency: {timing_stats.get('timing_consistency', 0):.2f} (0-1 scale, higher is better)

AUDIO QUALITY METRICS:
- Average Confidence: {audio_stats.get('average_confidence', 0):.2f} (0-1 scale)
- Min Confidence: {audio_stats.get('min_confidence', 0):.2f}
- Max Confidence: {audio_stats.get('max_confidence', 0):.2f}

DETAILED NOTE ANALYSIS:
"""
        
        # Add individual note details if available
        if audio_stats.get("waveform_metrics"):
            prompt += "\nINDIVIDUAL NOTE QUALITY:\n"
            for i, metrics in enumerate(audio_stats["waveform_metrics"]):
                if metrics:
                    prompt += f"- Note {i+1}: Sustain={metrics.get('sustain_duration', 0):.2f}s, "
                    prompt += f"Attack={metrics.get('attack_quality', 0):.2f}, "
                    prompt += f"Stability={metrics.get('frequency_stability', 0):.2f}, "
                    prompt += f"Noise={metrics.get('noise_level', 0):.2f}\n"
        
        # Add algorithm performance
        if audio_stats.get("algorithm_used"):
            prompt += f"\nDETECTION ALGORITHM: {', '.join(set(audio_stats['algorithm_used']))}\n"
        
        # Add specific analysis instructions
        prompt += """

ANALYSIS INSTRUCTIONS:
Provide exactly 3 sections with concise, musical feedback (no exact numbers):

1. TIMING FEEDBACK:
   • Comment on overall rhythm consistency
   • Identify if student is rushing or dragging
   • Give general timing improvement tips

2. AUDIO QUALITY FEEDBACK:
   • Note any buzzing, intonation, or technique issues
   • Comment on note sustain and attack quality
   • Reference specific notes by their position (first note, middle notes, etc.)

3. PRACTICE TIPS + ENCOURAGEMENT:
   • 2-3 specific practice exercises
   • 1 encouraging comment about what went well

IMPORTANT: 
- Give general musical advice, not exact millisecond targets
- Use musical terminology (rushing, dragging, buzzing, intonation)
- Be encouraging and constructive
- Keep each bullet point under 20 words
- Focus on overall patterns, not specific numbers

ANALYSIS:
"""
        
        return prompt
    
    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Send prompt to Ollama and return response."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused responses
                    "top_p": 0.8,        # Lower top_p for more focused responses
                    "max_tokens": 500,    # Reduced from 1000 for faster generation
                    "num_predict": 500,   # Explicit token limit
                    "repeat_penalty": 1.1, # Prevent repetitive responses
                    "top_k": 10          # Limit vocabulary choices
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=15  # Reduced timeout from 30s
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            return None
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Is it running?")
            return None
        except Exception as e:
            print(f"Unexpected error querying Ollama: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        return self.available_models.copy()
    
    def test_connection(self) -> bool:
        """Test if Ollama is accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False




def create_llm_analyzer() -> LLMAnalyzer:
    """Factory function to create LLM analyzer with error handling."""
    try:
        analyzer = LLMAnalyzer()
        if analyzer.test_connection():
            return analyzer
        else:
            st.warning("⚠️ Ollama not accessible. Please ensure Ollama is running.")
            return None
    except Exception as e:
        st.error(f"❌ Error creating LLM analyzer: {e}")
        return None
