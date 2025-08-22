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
Provide exactly 3 bullet points for each of the 3 sections. Use newlines for each bullet point.

PERFORMANCE DATA:
Scale: {perf_summary.get('scale_name', 'Unknown')}
Tempo: {perf_summary.get('tempo_bpm', 0)} BPM
Session Duration: {perf_summary.get('session_duration', 0):.1f}s
Notes Completed: {perf_summary.get('notes_completed', 0)}
Overall Rating: {perf_summary.get('overall_rating', 'Unknown')}

{self._get_scale_position_mapping(perf_summary.get('scale_name', 'Unknown'))}

DETAILED TIMING ANALYSIS:
{self._analyze_timing_patterns(data)}

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
Provide exactly 3 bullet points for each section. Use newlines for each bullet point.
Use the scale position mapping above to reference specific notes.

FOLLOW THIS EXACT FORMAT (adapt the content based on actual results):

1. TIMING FEEDBACK:
• [Specific note transition with timing issue] - [specific improvement tip]
• [Another specific transition issue] - [specific improvement tip]  
• [Overall timing improvement suggestion]

2. AUDIO QUALITY FEEDBACK:
• [Specific note with quality issue] - [specific technique tip]
• [Note with good quality] - [encouragement to maintain]
• [General quality improvement area] - [specific practice focus]

3. PRACTICE TIPS:
• [Specific exercise for identified timing issue]
• [Specific exercise for identified quality issue]
• [Encouraging comment about what went well]

ADAPTATION RULES:
- Use the scale position mapping (1:C, 2:D, 3:E, etc.) to reference notes
- If no timing issues found, focus on maintaining good timing
- If no quality issues found, focus on maintaining good technique
- Always provide 3 bullet points per section
- Keep each bullet point under 25 words
- Use musical terminology (rushing, dragging, buzzing, intonation, sustain, attack)
- Be specific about which notes have issues and which are good
- Use the detailed timing analysis provided above for specific feedback

EXAMPLE ADAPTATION:
If the data shows C to D was rushed by 0.15s and Note 3 (E) has buzzing:
• C to D transition was rushed - focus on maintaining steady tempo
• Note 3 (E) has slight buzzing - check finger placement
• Practice C to D transition slowly to fix rushing habit

If the data shows good timing but some quality issues:
• Excellent timing consistency throughout the scale
• Note 5 (G) shows good sustain quality - maintain this technique
• Work on finger independence exercises for buzzing issues

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

    def _get_scale_position_mapping(self, scale_name: str) -> str:
        """Create a 1-1 mapping of scale positions to note names."""
        scale_notes = self._get_scale_notes_from_name(scale_name)
        
        mapping = "SCALE POSITION MAPPING:\n"
        for i, note in enumerate(scale_notes):
            mapping += f"{i+1}: {note}\n"
        
        return mapping
    
    def _analyze_timing_patterns(self, data: Dict) -> str:
        """Analyze timing data and identify specific note transitions with delays."""
        try:
            timing_stats = data.get("timing_analysis", {})
            note_delays = timing_stats.get("note_delays", [])
            
            if len(note_delays) < 2:
                return "Insufficient timing data for analysis."
            
            # Get scale notes for reference
            scale_name = data.get("performance_summary", {}).get("scale_name", "")
            scale_notes = self._get_scale_notes_from_name(scale_name)
            
            # Find delays greater than threshold (e.g., 0.1s)
            delay_threshold = 0.1
            problematic_transitions = []
            
            for i, delay in enumerate(note_delays[1:], 1):  # Skip first note (no delay)
                if abs(delay) > delay_threshold:
                    if i < len(scale_notes) and i-1 < len(scale_notes):
                        from_note = scale_notes[i-1]
                        to_note = scale_notes[i]
                        transition_type = "rushed" if delay < 0 else "delayed"
                        problematic_transitions.append({
                            "from": from_note,
                            "to": to_note,
                            "delay": delay,
                            "type": transition_type
                        })
            
            # Format the timing analysis
            if problematic_transitions:
                timing_analysis = "TIMING ISSUES IDENTIFIED:\n"
                for transition in problematic_transitions:
                    timing_analysis += f"• {transition['from']} to {transition['to']}: {transition['type']} by {abs(transition['delay']):.2f}s\n"
                
                # Add overall timing summary
                avg_delay = timing_stats.get("average_delay", 0)
                consistency = timing_stats.get("timing_consistency", 0)
                
                timing_analysis += f"\nOVERALL TIMING:\n"
                timing_analysis += f"• Average delay: {avg_delay:.2f}s\n"
                timing_analysis += f"• Consistency score: {consistency:.2f} (0-1, higher is better)\n"
                
                return timing_analysis
            else:
                return "TIMING ANALYSIS: All note transitions were within acceptable timing range."
                
        except Exception as e:
            print(f"Error analyzing timing patterns: {e}")
            return "Error analyzing timing patterns."
    
    def _get_scale_notes_from_name(self, scale_name: str) -> List[str]:
        """Get scale notes from scale name."""
        scale_notes = {
            "C Major Scale": ["C", "D", "E", "F", "G", "A", "B", "C"],
            "G Major Scale": ["G", "A", "B", "C", "D", "E", "F#", "G"],
            "D Major Scale": ["D", "E", "F#", "G", "A", "B", "C#", "D"],
            "A Minor Scale": ["A", "B", "C", "D", "E", "F", "G", "A"],
            "E Minor Scale": ["E", "F#", "G", "A", "B", "C", "D", "E"]
        }
        return scale_notes.get(scale_name, ["C", "D", "E", "F", "G", "A", "B", "C"])




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
