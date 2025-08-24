"""Main Streamlit application for the Agentic Music Tutor."""

import streamlit as st
import time
from typing import List

# Import our modular components from the new package structure
from src.data.config import (
    SAMPLE_RATE, CHANNELS, HOP_SIZE, BUFFER_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CHUNK_DURATION,
    DEFAULT_ALGORITHM_MODE, ALGORITHM_MODES,
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    DEFAULT_TEMPO_BPM, TEMPO_MIN_BPM, TEMPO_MAX_BPM,
    NOTE_TIMING_TOLERANCE
)
from src.utils.utils import load_ground_truth, calculate_accuracy
from src.core.note_detector import detect_note_from_frame
from src.core.audio_processor import record_audio_chunk, get_audio_stats, analyze_waveform_quality
from src.utils.visualizations import create_waveform_plot, create_note_display_plot
from src.ui.styles import MAIN_STYLES, get_note_display_html
from src.core.scale_trainer import ScaleTrainer
from src.core.llm_analyzer import create_llm_analyzer

# Import new agentic components
from src.core.music_agent import create_music_agent
from src.core.conversation_manager import create_conversation_manager
from src.core.practice_orchestrator import create_practice_orchestrator

# Import required libraries with error handling
try:
    import sounddevice as sd
except Exception:
    st.error("sounddevice not available. Install with: pip install sounddevice")
    st.stop()

try:
    import aubio
except Exception:
    st.error("aubio not available. Install with: pip install aubio")
    st.stop()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = []
    if 'detected_notes' not in st.session_state:
        st.session_state.detected_notes = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'last_note_time' not in st.session_state:
        st.session_state.last_note_time = 0
    if 'scale_trainer' not in st.session_state:
        st.session_state.scale_trainer = ScaleTrainer()
    if 'llm_analyzer' not in st.session_state:
        st.session_state.llm_analyzer = create_llm_analyzer()
    if 'llm_analysis' not in st.session_state:
        st.session_state.llm_analysis = None
    
    # Initialize agentic components
    if 'music_agent' not in st.session_state:
        st.session_state.music_agent = create_music_agent()
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = create_conversation_manager(st.session_state.music_agent) if st.session_state.music_agent else None
    if 'practice_orchestrator' not in st.session_state:
        st.session_state.practice_orchestrator = create_practice_orchestrator(
            st.session_state.music_agent, 
            st.session_state.conversation_manager
        ) if st.session_state.music_agent and st.session_state.conversation_manager else None
    
    # Agentic mode state
    if 'agentic_mode' not in st.session_state:
        st.session_state.agentic_mode = False
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # AI tutor UI state
    if 'selected_scale' not in st.session_state:
        st.session_state.selected_scale = "C Major Scale"
    if 'selected_tempo' not in st.session_state:
        st.session_state.selected_tempo = 80


def render_recording_controls():
    """Render the recording control buttons."""
    col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
    
    with col1_1:
        if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
            st.session_state.recording = True
            st.session_state.audio_data = []
            st.session_state.detected_notes = []
            st.session_state.start_time = time.time()
            st.session_state.last_note_time = 0
    
    with col1_2:
        if st.button("⏹️ Stop Recording", use_container_width=True):
            st.session_state.recording = False
    
    with col1_3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.audio_data = []
            st.session_state.detected_notes = []
            st.session_state.start_time = None
            st.session_state.last_note_time = 0


def render_recording_status():
    """Render the current recording status."""
    if st.session_state.recording:
        st.success("🔴 Recording...")
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.write(f"⏱️ Recording time: {elapsed:.1f}s")
    else:
        st.info("⏸️ Ready to record")


def render_detected_notes():
    """Render the detected notes display."""
    if st.session_state.detected_notes:
        st.subheader("🎵 Detected Notes")
        note_text = " → ".join(st.session_state.detected_notes[-5:])  # Show last 5 notes
        st.markdown(get_note_display_html(note_text), unsafe_allow_html=True)


def render_statistics(expected_notes: List[str]):
    """Render the statistics and metrics."""
    st.header("📊 Statistics")
    
    if st.session_state.detected_notes:
        total_detected, correct_notes, accuracy = calculate_accuracy(
            st.session_state.detected_notes, expected_notes
        )
        
        st.metric("Total Notes", total_detected)
        st.metric("Correct Notes", correct_notes)
        st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Progress bar
        st.progress(accuracy / 100)
    else:
        st.write("No notes detected yet")


def render_visualizations():
    """Render the waveform and note comparison charts."""
    # Waveform display
    if st.session_state.audio_data:
        st.header("📈 Audio Waveform")
        audio_array = st.session_state.audio_data
        fig = create_waveform_plot(audio_array, SAMPLE_RATE, "Live Audio Recording")
        st.plotly_chart(fig, use_container_width=True)
    
    # Note comparison
    if st.session_state.detected_notes:
        st.header("🎼 Note Analysis")
        song_name, expected_notes = load_ground_truth()
        fig = create_note_display_plot(st.session_state.detected_notes, expected_notes)
        st.plotly_chart(fig, use_container_width=True)


def render_scale_training_progress():
    """Render the scale training progress and timing analysis."""
    progress = st.session_state.scale_trainer.get_training_progress()
    
    if progress["status"] == "training":
        st.header("🎼 Scale Training Progress")
        
        # Progress bar
        st.progress(progress["progress_percentage"] / 100)
        st.write(f"**Progress**: {progress['completed_notes']}/{progress['total_notes']} notes ({progress['progress_percentage']:.1f}%)")
        
        # Current status
        if progress["next_expected_note"]:
            st.info(f"**Next Note**: {progress['next_expected_note']} at {progress['next_expected_time']:.2f}s")
        
        # Timing accuracy
        if progress["average_delay"] > 0:
            st.metric("Average Delay", f"{progress['average_delay']:.2f}s")
        else:
            st.metric("Average Delay", "N/A")
        
        # Scale visualization
        scale_info = st.session_state.scale_trainer.get_scale_info()
        notes = scale_info["notes"]
        
        # Create a visual representation of the scale
        st.subheader("Scale Progress")
        for i, note in enumerate(notes):
            if i < progress["current_note_index"]:
                st.success(f"✅ {note}")  # Completed
            elif i == progress["current_note_index"]:
                st.info(f"🎯 {note}")    # Current
            else:
                st.write(f"⏳ {note}")    # Pending
    else:
        st.header("🎼 Scale Training")
        st.info("Click 'Start Training' to begin the C Major scale exercise")


def render_scale_visualization():
    """Render the scale visualization with single display and metronome."""
    st.subheader("🎼 Scale Progress")
    
    # Single scale display
    scale_info = st.session_state.scale_trainer.get_scale_info()
    notes = scale_info["notes"]
    
    # Create single scale visualization
    scale_html = ""
    for i, note in enumerate(notes):
        # Check if this note has been detected
        note_detected = False
        note_timing = ""
        note_delay = ""
        
        if st.session_state.detected_notes and i < len(st.session_state.detected_notes):
            note_detected = True
            # Get timing info for this note
            if st.session_state.scale_trainer.is_training and st.session_state.scale_trainer.start_time:
                if i == 0:
                    note_timing = "Expected: 0.0s"
                    note_delay = "Delay: 0.0s"
                else:
                    # For note-by-note delay calculation
                    if i < len(st.session_state.scale_trainer.note_delays):
                        # Get the delay for this specific note transition
                        note_timing = f"Expected: {i * (60.0 / st.session_state.scale_trainer.tempo_bpm):.1f}s"
                        
                        # Get the delay directly from the note_delays list
                        delay = st.session_state.scale_trainer.note_delays[i]
                        note_delay = f"Delay: {delay:+.1f}s"
                    else:
                        note_timing = f"Expected: {i * (60.0 / st.session_state.scale_trainer.tempo_bpm):.1f}s"
                        note_delay = "Delay: --"
            else:
                # Fallback for when training state is unclear
                note_timing = "Expected: --"
                note_delay = "Delay: --"
        else:
            # Undetected notes - show expected time when training
            if st.session_state.scale_trainer.is_training or (st.session_state.detected_notes and len(st.session_state.detected_notes) > 0):
                if i == 0:
                    note_timing = "Expected: 0.0s"
                    note_delay = "Delay: --"
                else:
                    expected_time = i * (60.0 / st.session_state.scale_trainer.tempo_bpm)
                    note_timing = f"Expected: {expected_time:.1f}s"
                    note_delay = "Delay: --"
            else:
                note_timing = "Expected: --"
                note_delay = "Delay: --"
        
        # Determine card color and timing info
        if note_detected:
            # Green card for detected notes
            card_html = f'<div style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #e8f5e8; border: 2px solid #4caf50; border-radius: 8px; text-align: center; min-width: 80px;"><div style="font-weight: bold; color: #2e7d32; margin-bottom: 0.5rem;">{note}</div><div style="font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;">{note_timing}</div><div style="font-size: 0.8rem; color: #666;">{note_delay}</div></div>'
        else:
            # Bland color for undetected notes
            card_html = f'<div style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #f5f5f5; border: 2px solid #e0e0e0; border-radius: 8px; text-align: center; min-width: 80px;"><div style="font-weight: bold; color: #757575; margin-bottom: 0.5rem;">{note}</div><div style="font-size: 0.8rem; color: #999; margin-bottom: 0.2rem;">{note_timing}</div><div style="font-size: 0.8rem; color: #999;">{note_delay}</div></div>'
        
        scale_html += card_html
    
    st.markdown(scale_html, unsafe_allow_html=True)


def process_audio_chunk(chunk_duration: float, confidence_threshold: float, algorithm_mode: str):
    """Process a single audio chunk for real-time note detection."""
    # Record a small audio chunk
    audio_chunk = record_audio_chunk(chunk_duration)
    
    if audio_chunk is not None:
        # Add to audio data for waveform
        st.session_state.audio_data.extend(audio_chunk)
        
        # Detect note from this chunk
        detection_result = detect_note_from_frame(audio_chunk, confidence_threshold, algorithm_mode)
        
        if detection_result and detection_result[0]:  # Check if note was detected
            detected_note, algorithm_used, confidence_score = detection_result
            current_time = time.time()
            # Only add note if it's different from last one and enough time has passed
            if (not st.session_state.detected_notes or 
                detected_note != st.session_state.detected_notes[-1] or
                current_time - st.session_state.last_note_time > 0.5):  # 500ms minimum between notes
                
                # For scale training, check if this is the correct next note
                if st.session_state.scale_trainer.is_training:
                    expected_note = st.session_state.scale_trainer.get_expected_note()
                    if detected_note == expected_note:
                        # Correct note - add to detected notes and update training
                        st.session_state.detected_notes.append(detected_note)
                        st.session_state.last_note_time = current_time
                        st.success(f"🎵 Correct note detected: {detected_note} (via {algorithm_used}, conf: {confidence_score:.3f})")
                        
                        # Update scale trainer with enhanced data
                        waveform_data = analyze_waveform_quality(audio_chunk)
                        timing_result = st.session_state.scale_trainer.check_note_timing(
                            detected_note, 
                            confidence_score, 
                            waveform_data, 
                            algorithm_used, 
                            audio_chunk
                        )
                        if timing_result[0]:  # Note was processed
                            timing_error, expected_time = timing_result[1], timing_result[2]
                            if timing_error > NOTE_TIMING_TOLERANCE:
                                st.warning(f"⏰ Note correct but timing off by {timing_error:.2f}s")
                    else:
                        # Wrong note - show warning but don't progress
                        st.warning(f"❌ Wrong note detected: {detected_note}. Expected: {expected_note}")
                else:
                    # Free play mode - add any detected note
                    st.session_state.detected_notes.append(detected_note)
                    st.session_state.last_note_time = current_time
                    st.success(f"🎵 New note detected: {detected_note} (via {algorithm_used}, conf: {confidence_score:.3f})")
        else:
            detected_note = None
        
        # Debug info
        audio_stats = get_audio_stats(audio_chunk)
        st.write(f"Audio chunk recorded: {audio_stats.get('samples', 0)} samples")
        st.write(f"Audio range: {audio_stats.get('min_amplitude', 0):.4f} to {audio_stats.get('max_amplitude', 0):.4f}")
        st.write(f"Processing {len(audio_chunk) // HOP_SIZE} frames with buffer size {BUFFER_SIZE}")
        if detection_result and detection_result[0]:
            st.write(f"Note detected: {detection_result[0]}")
        else:
            st.write("No note detected in this chunk")


def main():
    """Main application function."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom CSS for navbar and layout
    st.markdown(MAIN_STYLES, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check if user has selected a mode
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = None
    
    # Landing page - show mode selection
    if st.session_state.app_mode is None:
        render_landing_page()
    elif st.session_state.app_mode == 'note_detection':
        render_note_detection_mode()
    elif st.session_state.app_mode == 'scale_training':
        render_scale_training_mode()
    elif st.session_state.app_mode == 'ai_tutor':
        render_ai_tutor_mode()
    
    # Settings button at the bottom of the page
    render_bottom_settings()


def render_landing_page():
    """Render the compact, centered landing page."""
    # Title Board with the original purplish color
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border: 2px solid #dee2e6; border-radius: 15px; padding: 3rem 2rem; 
                margin: 2rem 0; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin-bottom: 1rem; font-size: 2.5rem;">🎸 Agentic Music Tutor</h1>
        <h2 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">Your Musical Journey Starts Here</h2>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.9); line-height: 1.6; max-width: 500px; margin: 0 auto;">
            Choose your learning path below and start developing your musical skills with AI-powered guidance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pathway Cards and Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Free Play Card
        st.markdown("""
        <div style="background: white; border: 2px solid #e0e0e0; border-radius: 15px; 
                    padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #28a745; margin-bottom: 1rem; font-size: 1.4rem;">🎵 Free Play Mode</h3>
            <p style="color: #555; line-height: 1.5; margin-bottom: 1.5rem;">
                Practice note detection and pitch recognition without constraints. 
                Perfect for exploring your instrument and improving your ear training.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Button with same width as card
        if st.button("🎵 Start Free Play", key="free_play_btn", use_container_width=False):
            st.session_state.app_mode = 'note_detection'
            st.rerun()
    
    with col2:
        # Scale Training Card
        st.markdown("""
        <div style="background: white; border: 2px solid #e0e0e0; border-radius: 15px; 
                    padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #007bff; margin-bottom: 1rem; font-size: 1.4rem;">🎼 Scale Training</h3>
            <p style="color: #555; line-height: 1.5; margin-bottom: 1.5rem;">
                Learn major and minor scales with tempo-based timing analysis. 
                Choose from C Major, G Major, D Major, A Minor, and E Minor scales.
                Develop rhythm, timing, and scale proficiency with real-time feedback.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Button with same width as card
        if st.button("🎼 Start Scale Training", key="scale_training_btn", use_container_width=False):
            st.session_state.app_mode = 'scale_training'
            st.rerun()
    
    with col3:
        # Agentic AI Tutor Card
        st.markdown("""
        <div style="background: white; border: 2px solid #e0e0e0; border-radius: 15px; 
                    padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #9c27b0; margin-bottom: 1rem; font-size: 1.4rem;">🤖 AI Agentic Tutor</h3>
            <p style="color: #555; line-height: 1.5; margin-bottom: 1.5rem;">
                Experience autonomous music tutoring with AI that adapts to your skill level,
                sets personalized goals, and guides your entire learning journey.
                Natural language interaction and intelligent practice planning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Button with same width as card
        if st.button("🤖 Start AI Tutor", key="ai_tutor_btn", use_container_width=False):
            st.session_state.app_mode = 'ai_tutor'
            st.rerun()


def render_note_detection_mode():
    """Render the free play note detection mode."""
    st.header("🎵 Free Play Mode")
    
    # Back button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Landing", key="back_note_detection"):
            st.session_state.app_mode = None
            st.rerun()
    
    # Load ground truth
    song_name, expected_notes = load_ground_truth()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎙️ Recording")
        
        # Recording controls
        render_recording_controls()
        
        # Status display
        render_recording_status()
        
        # Real-time note detection display
        render_detected_notes()
    
    with col2:
        render_statistics(expected_notes)
    
    # Visualizations
    render_visualizations()
    
    # Real-time audio capture and note detection
    if st.session_state.recording:
        process_audio_chunk(
            DEFAULT_CHUNK_DURATION, 
            DEFAULT_CONFIDENCE_THRESHOLD, 
            DEFAULT_ALGORITHM_MODE
        )
        st.rerun()


def render_scale_training_mode():
    """Render the scale training mode with integrated settings."""
    st.header("🎼 Scale Training Mode")
    
    # Back button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Landing", key="back_scale_training"):
            st.session_state.app_mode = None
            st.rerun()
    
    # Scale training settings (moved from sidebar to main page)
    st.subheader("⚙️ Training Settings")
    
    # Scale selector dropdown
    available_scales = st.session_state.scale_trainer.get_available_scales()
    scale_names = [scale['name'] for scale in available_scales]
    
    if scale_names:
        selected_scale = st.selectbox(
            "Select Scale:",
            scale_names,
            index=0,  # Default to first scale
            key="scale_selector"
        )
        
        # Load selected scale if different from current
        current_scale = st.session_state.scale_trainer.scale_data.get('scale_name', '')
        if selected_scale != current_scale:
            if st.session_state.scale_trainer.load_scale_by_name(selected_scale):
                st.success(f"Loaded {selected_scale}")
                st.rerun()
            else:
                st.error(f"Failed to load {selected_scale}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scale_info = st.session_state.scale_trainer.get_scale_info()
        st.write(f"**Scale**: {scale_info['name']}")
        st.write(f"**Notes**: {' → '.join(scale_info['notes'])}")
    
    with col2:
        # Tempo control
        tempo_bpm = st.slider(
            "Tempo (BPM)",
            TEMPO_MIN_BPM, TEMPO_MAX_BPM,
            st.session_state.scale_trainer.tempo_bpm,
            5
        )
        
        # Update tempo when changed
        if tempo_bpm != st.session_state.scale_trainer.tempo_bpm:
            st.session_state.scale_trainer.set_tempo(tempo_bpm)
    
    with col3:
        # Training controls
        scale_name = st.session_state.scale_trainer.scale_data.get('scale_name', 'Scale')
        if st.button(f"▶️ Start {scale_name} Training", use_container_width=True, type="primary"):
            st.session_state.scale_trainer.start_training()
            st.session_state.recording = True  # Start recording
            st.rerun()
        
        if st.button("⏹️ Stop Training", use_container_width=True):
            st.session_state.recording = False  # Stop recording
            st.session_state.scale_trainer.stop_training()  # Actually stop training
            st.session_state.llm_analysis = None  # Clear previous analysis
            st.rerun()
        
        if st.button("🔄 Restart Training", use_container_width=True):
            st.session_state.scale_trainer.stop_training()
            st.session_state.recording = False
            st.session_state.audio_data = []
            st.session_state.detected_notes = []
            st.session_state.start_time = None
            st.session_state.last_note_time = 0
            st.session_state.llm_analysis = None  # Clear previous analysis
            st.rerun()
        

    
    # Scale training progress
    # render_scale_training_progress()  # Removed - replaced with new visual system
    
    # Visual scale progress with time-based positioning
    render_scale_visualization()
    
    # LLM Analysis Section
    render_llm_analysis_section()
    
    # Recording status (only show when training)
    if st.session_state.scale_trainer.is_training:
        st.subheader("🎙️ Training Session")
        render_recording_status()
        render_detected_notes()
        
        # Real-time audio capture and note detection
        if st.session_state.recording:
            process_audio_chunk(
                DEFAULT_CHUNK_DURATION, 
                DEFAULT_CONFIDENCE_THRESHOLD, 
                DEFAULT_ALGORITHM_MODE
            )
            st.rerun()


def render_llm_analysis_section():
    """Render the LLM analysis section with AI-powered insights."""
    st.markdown("---")
    st.subheader("🤖 AI Performance Analysis")
    
    # Check if we have a completed training session
    if len(st.session_state.scale_trainer.actual_timings) == 0:
        st.info("Complete a training session to get AI-powered analysis!")
        return
    
    # Check if LLM analyzer is available
    if not st.session_state.llm_analyzer:
        st.warning("⚠️ LLM analysis not available. Please ensure Ollama is running.")
        return
    
    # Get comprehensive analysis data
    analysis_data = st.session_state.scale_trainer.get_comprehensive_analysis_data()
    
    if not analysis_data:
        st.info("No performance data available for analysis.")
        return
    
    # Display performance summary
    perf_summary = analysis_data["performance_summary"]
    timing_stats = analysis_data["timing_analysis"]
    audio_stats = analysis_data["audio_quality"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Rating", perf_summary["overall_rating"])
    with col2:
        st.metric("Timing Consistency", f"{timing_stats['timing_consistency']:.2f}")
    with col3:
        st.metric("Avg Confidence", f"{audio_stats['average_confidence']:.2f}")
    with col4:
        st.metric("Session Duration", f"{perf_summary['session_duration']:.1f}s")
    
    # LLM Analysis Button
    if st.session_state.scale_trainer.is_training:
        st.info("⏸️ Complete your training session to get AI analysis")
    else:
        if st.button("🧠 Get AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing your performance..."):
                # Get LLM analysis
                llm_analysis = st.session_state.llm_analyzer.analyze_performance(analysis_data)
                if llm_analysis:
                    st.session_state.llm_analysis = llm_analysis
                    st.rerun()
                else:
                    st.error("Failed to get AI analysis. Please check Ollama connection.")
    
    # Display LLM analysis if available
    if st.session_state.llm_analysis:
        st.markdown("### 📊 AI Performance Insights")
        st.markdown("---")
        
        # Parse and display the analysis in structured sections
        analysis_text = st.session_state.llm_analysis
        
        # Try to parse the sections
        if "1. TIMING FEEDBACK:" in analysis_text and "2. AUDIO QUALITY FEEDBACK:" in analysis_text and "3. PRACTICE TIPS:" in analysis_text:
            # Split into sections
            sections = analysis_text.split("1. TIMING FEEDBACK:")
            if len(sections) > 1:
                timing_section = sections[1].split("2. AUDIO QUALITY FEEDBACK:")[0].strip()
                audio_section = sections[1].split("2. AUDIO QUALITY FEEDBACK:")[1].split("3. PRACTICE TIPS:")[0].strip()
                practice_section = sections[1].split("3. PRACTICE TIPS:")[1].strip()
                
                # Display in structured format - truly side-by-side with compact cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**⏱️ Timing Feedback**")
                    # Create a compact card-like display
                    timing_clean = timing_section.replace("1. TIMING FEEDBACK:", "").strip()
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    {timing_clean}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**🎵 Audio Quality Feedback**")
                    audio_clean = audio_section.replace("2. AUDIO QUALITY FEEDBACK:", "").strip()
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    {audio_clean}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**💡 Practice Tips**")
                    practice_clean = practice_section.replace("3. PRACTICE TIPS:", "").strip()
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    {practice_clean}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Fallback to original display if parsing fails
            st.markdown(analysis_text)
        
        # Add a button to regenerate analysis
        if st.button("🔄 Regenerate Analysis", use_container_width=False):
            st.session_state.llm_analysis = None
            st.rerun()


def render_ai_tutor_mode():
    """Render the AI agentic tutor mode with integrated scale training."""
    st.header("🤖 AI Agentic Music Tutor")
    
    # Back button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Landing", key="back_ai_tutor"):
            st.session_state.app_mode = None
            st.rerun()
    
    # Check if agentic components are available
    if not st.session_state.music_agent or not st.session_state.conversation_manager:
        st.error("❌ AI Tutor components not available. Please ensure Ollama is running.")
        st.info("💡 Make sure you have Ollama installed and running with the llama3.2:3b model.")
        return
    
    # Main content - AI Tutor on the right, Scale Training on the left
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎼 AI-Guided Scale Training")
        
        # Scale training settings
        st.subheader("⚙️ Training Settings")
        
        # Scale selector dropdown
        available_scales = st.session_state.scale_trainer.get_available_scales()
        scale_names = [scale['name'] for scale in available_scales]
        
        if scale_names:
            # Initialize selected scale from session state or current scale
            if 'selected_scale' not in st.session_state:
                st.session_state.selected_scale = scale_names[0]
            
            selected_scale = st.selectbox(
                "Select Scale:",
                scale_names,
                index=scale_names.index(st.session_state.selected_scale) if st.session_state.selected_scale in scale_names else 0,
                key="ai_scale_selector"
            )
            
            # Update session state when user changes selection
            if selected_scale != st.session_state.selected_scale:
                st.session_state.selected_scale = selected_scale
            
            # Load selected scale if different from current
            current_scale = st.session_state.scale_trainer.scale_data.get('scale_name', '')
            if selected_scale != current_scale:
                if st.session_state.scale_trainer.load_scale_by_name(selected_scale):
                    st.success(f"Loaded {selected_scale}")
                    st.rerun()
                else:
                    st.error(f"Failed to load {selected_scale}")
        
        # Scale info and controls
        col1a, col2a, col3a = st.columns(3)
        
        with col1a:
            scale_info = st.session_state.scale_trainer.get_scale_info()
            st.write(f"**Scale**: {scale_info['name']}")
            st.write(f"**Notes**: {' → '.join(scale_info['notes'])}")
        
        with col2a:
            # Tempo control
            # Initialize selected tempo from session state or current tempo
            if 'selected_tempo' not in st.session_state:
                st.session_state.selected_tempo = st.session_state.scale_trainer.tempo_bpm
            
            tempo_bpm = st.slider(
                "Tempo (BPM)",
                TEMPO_MIN_BPM, TEMPO_MAX_BPM,
                st.session_state.selected_tempo,
                5,
                key="ai_tempo_slider"
            )
            
            # Update session state when user changes tempo
            if tempo_bpm != st.session_state.selected_tempo:
                st.session_state.selected_tempo = tempo_bpm
            
            # Update tempo when changed
            if tempo_bpm != st.session_state.scale_trainer.tempo_bpm:
                st.session_state.scale_trainer.set_tempo(tempo_bpm)
        
        with col3a:
            # Training controls
            scale_name = st.session_state.scale_trainer.scale_data.get('scale_name', 'Scale')
            if st.button(f"▶️ Start {scale_name} Training", use_container_width=True, type="primary", key="ai_start_training"):
                st.session_state.scale_trainer.start_training()
                st.session_state.recording = True
                st.rerun()
            
            if st.button("⏹️ Stop Training", use_container_width=True, key="ai_stop_training"):
                st.session_state.recording = False
                st.session_state.scale_trainer.stop_training()
                st.rerun()
            
            if st.button("🔄 Restart Training", use_container_width=True, key="ai_restart_training"):
                st.session_state.scale_trainer.stop_training()
                st.session_state.recording = False
                st.session_state.audio_data = []
                st.session_state.detected_notes = []
                st.session_state.start_time = None
                st.session_state.last_note_time = 0
                st.rerun()
        
        # Visual scale progress
        render_scale_visualization()
        
        # Recording status and note detection (only show when training)
        if st.session_state.scale_trainer.is_training:
            st.subheader("🎙️ Training Session")
            render_recording_status()
            render_detected_notes()
            
            # Real-time audio capture and note detection
            if st.session_state.recording:
                process_audio_chunk(
                    DEFAULT_CHUNK_DURATION, 
                    DEFAULT_CONFIDENCE_THRESHOLD, 
                    DEFAULT_ALGORITHM_MODE
                )
                st.rerun()
    
    with col2:
        st.subheader("🤖 AI Tutor Assistant")
        
        # Student profile display
        if st.session_state.music_agent:
            profile = st.session_state.music_agent.student_profile
            st.info(f"""
            **Student Profile:**
            - **Level:** {profile.level.value.title()}
            - **Instrument:** {profile.preferred_instrument.title()}
            - **Practice Time:** {profile.practice_time_per_day} minutes/day
            - **Completed Scales:** {len(profile.completed_scales)}
            - **Current Goals:** {len(profile.current_goals)}
            """)
        
        # Practice session status
        st.subheader("🎯 Session Status")
        
        if st.session_state.practice_orchestrator and st.session_state.practice_orchestrator.has_active_session():
            # Active session
            session = st.session_state.practice_orchestrator.current_session
            st.success(f"**Active:** {session.current_scale}")
            st.info(f"**Phase:** {session.current_phase.value.replace('_', ' ').title()}")
            st.write(f"**Focus:** {', '.join([area.value for area in session.focus_areas])}")
            
            # Session guidance
            st.markdown("**🤖 AI Guidance:**")
            st.write(session.agent_guidance[-1] if session.agent_guidance else "No guidance available")
            
            # AI Performance Analysis (moved here from main UI)
            if st.session_state.scale_trainer.actual_timings and len(st.session_state.scale_trainer.actual_timings) > 0:
                st.subheader("📊 AI Performance Analysis")
                
                # Get LLM analysis
                if st.session_state.llm_analyzer:
                    try:
                        # Prepare performance data for analysis
                        performance_data = {
                            "scale_name": st.session_state.scale_trainer.scale_data.get("scale_name", "Unknown"),
                            "tempo_bpm": st.session_state.scale_trainer.tempo_bpm,
                            "actual_timings": st.session_state.scale_trainer.actual_timings,
                            "expected_timings": st.session_state.scale_trainer.expected_timings,
                            "detected_notes": st.session_state.detected_notes[-10:] if st.session_state.detected_notes else []
                        }
                        
                        # Get AI analysis
                        analysis = st.session_state.llm_analyzer.analyze_performance(performance_data)
                        st.markdown(analysis)
                        
                    except Exception as e:
                        st.error(f"Error getting AI analysis: {str(e)}")
                else:
                    st.info("AI analyzer not available for performance insights.")
            
            # Session controls
            if st.button("⏹️ End Session", key="ai_end_session", use_container_width=True):
                summary = st.session_state.practice_orchestrator.end_session()
                st.session_state.recording = False
                if "error" not in summary:
                    st.success("Session ended!")
                else:
                    st.warning(summary["error"])
                st.rerun()
            
            if st.button("🔄 Restart Session", key="ai_restart_session", use_container_width=True):
                st.session_state.practice_orchestrator.end_session()
                st.rerun()
        else:
            # Start new session
            st.info("Ready to start a new practice session!")
            
            student_input = st.text_area(
                "What would you like to work on?",
                placeholder="e.g., I want to practice C Major scale at 80 BPM, I want to improve my timing on G Major scale, etc.",
                height=80,
                key="ai_student_input"
            )
            
            if st.button("🚀 Start AI Session", key="ai_start_session", type="primary", use_container_width=True):
                if st.session_state.practice_orchestrator:
                    # Start the AI-guided session
                    session = st.session_state.practice_orchestrator.start_autonomous_session(student_input)
                    if session:
                        # Auto-update the scale and tempo based on AI session
                        if session.current_scale:
                            # Find and select the scale in the dropdown
                            available_scales = st.session_state.scale_trainer.get_available_scales()
                            scale_names = [scale['name'] for scale in available_scales]
                            if session.current_scale in scale_names:
                                # Update the scale trainer and session state
                                st.session_state.scale_trainer.load_scale_by_name(session.current_scale)
                                # Store the selected scale for the dropdown
                                st.session_state.selected_scale = session.current_scale
                                st.success(f"🎯 AI selected scale: {session.current_scale}")
                        
                        # Update tempo
                        if session.tempo_bpm:
                            st.session_state.scale_trainer.set_tempo(session.tempo_bpm)
                            # Store the tempo for the slider
                            st.session_state.selected_tempo = session.tempo_bpm
                            st.success(f"🎵 AI set tempo: {session.tempo_bpm} BPM")
                        
                        # Auto-start recording
                        st.session_state.recording = True
                        st.session_state.scale_trainer.start_training()
                        
                        st.success(f"🚀 Started {session.current_scale} practice at {session.tempo_bpm} BPM!")
                        st.rerun()
                    else:
                        st.error("Failed to start session. Please try again.")
                else:
                    st.error("Practice orchestrator not available.")


def render_bottom_settings():
    """Render the settings button at the bottom of the page."""
    st.markdown("---")  # Separator line
    
    # Settings button at the bottom
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("⚙️ Settings", key="settings_toggle", help="Settings", use_container_width=False):
            st.session_state.show_settings = not st.session_state.get('show_settings', False)
            st.rerun()
    
    # Settings panel
    if st.session_state.get('show_settings', False):
        st.markdown("""
        <div style="background: white; border: 1px solid #ccc; border-radius: 10px; 
                    padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        """, unsafe_allow_html=True)
        
        st.subheader("⚙️ Settings")
        
        # Detection parameters
        st.write("**Detection Parameters**")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 1.0, 
            DEFAULT_CONFIDENCE_THRESHOLD, 
            0.05,
            key="settings_confidence"
        )
        chunk_duration = st.slider(
            "Chunk Duration (seconds)", 
            0.1, 2.0, 
            DEFAULT_CHUNK_DURATION, 
            0.1,
            key="settings_chunk"
        )
        
        # Algorithm selection
        st.write("**Pitch Detection Algorithm**")
        algorithm_mode = st.selectbox(
            "Algorithm Mode",
            ALGORITHM_MODES,
            index=0,
            key="settings_algorithm"
        )
        
        # Audio settings info
        st.write("**Audio Settings**")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Channels: {CHANNELS}")
        st.write(f"Hop Size: {HOP_SIZE}")
        st.write(f"Buffer Size: {BUFFER_SIZE}")
        
        # Close button
        if st.button("✕ Close", key="close_settings"):
            st.session_state.show_settings = False
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
