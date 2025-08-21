# Configuration constants for the Agentic Music Tutor

# Audio Configuration
SAMPLE_RATE = 48000
CHANNELS = 2
HOP_SIZE = 256
BUFFER_SIZE = 256  # Must match HOP_SIZE for aubio compatibility

# Note Configuration
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Default Values
DEFAULT_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_CHUNK_DURATION = 0.5
DEFAULT_ALGORITHM_MODE = "Multi-Algorithm (Best)"

# Algorithm Options
ALGORITHM_MODES = [
    "Multi-Algorithm (Best)",
    "Yin Only", 
    "YinFFT Only", 
    "MComb Only", 
    "Schmitt Only"
]

# File Paths
GROUND_TRUTH_PATH = "ground_truth.json"

# UI Configuration
PAGE_TITLE = "🎸 Agentic Music Tutor"
PAGE_ICON = "🎸"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
