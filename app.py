"""Main Streamlit application for the Agentic Music Tutor."""

import streamlit as st
import time
from typing import List

# Import our modular components
from config import (
    SAMPLE_RATE, CHANNELS, HOP_SIZE, BUFFER_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CHUNK_DURATION,
    DEFAULT_ALGORITHM_MODE, ALGORITHM_MODES,
    PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE
)
from utils import load_ground_truth, calculate_accuracy
from note_detector import detect_note_from_frame
from audio_processor import record_audio_chunk, get_audio_stats
from visualizations import create_waveform_plot, create_note_display_plot
from styles import MAIN_STYLES, get_main_header_html, get_note_display_html

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


def render_sidebar(song_name: str, expected_notes: List[str]):
    """Render the sidebar with settings and parameters."""
    with st.sidebar:
        st.header("🎵 Settings")
        st.write(f"**Song**: {song_name}")
        st.write(f"**Expected**: {', '.join(expected_notes)}")
        
        st.header("🎚️ Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 1.0, 
            DEFAULT_CONFIDENCE_THRESHOLD, 
            0.05
        )
        chunk_duration = st.slider(
            "Chunk Duration (seconds)", 
            0.1, 2.0, 
            DEFAULT_CHUNK_DURATION, 
            0.1
        )
        
        st.subheader("🎵 Pitch Detection Algorithm")
        algorithm_mode = st.selectbox(
            "Algorithm Mode",
            ALGORITHM_MODES,
            index=0
        )
        
        st.header("📊 Audio Settings")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Channels: {CHANNELS}")
        st.write(f"Hop Size: {HOP_SIZE}")
        st.write(f"Buffer Size: {BUFFER_SIZE}")
        
        return confidence_threshold, chunk_duration, algorithm_mode


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
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE
    )
    
    # Apply custom CSS
    st.markdown(MAIN_STYLES, unsafe_allow_html=True)
    
    # Header
    st.markdown(get_main_header_html(), unsafe_allow_html=True)
    
    # Load ground truth
    song_name, expected_notes = load_ground_truth()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    confidence_threshold, chunk_duration, algorithm_mode = render_sidebar(song_name, expected_notes)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎙️ Recording")
        
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
        process_audio_chunk(chunk_duration, confidence_threshold, algorithm_mode)
        
        # Force rerun for real-time updates
        st.rerun()


if __name__ == "__main__":
    main()
