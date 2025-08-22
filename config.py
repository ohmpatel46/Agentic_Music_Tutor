# Configuration constants for the Agentic Music Tutor

# Audio Configuration
SAMPLE_RATE = 48000
CHANNELS = 2
HOP_SIZE = 256
BUFFER_SIZE = 256  # Must match HOP_SIZE for aubio compatibility

# Note Configuration
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Default Values
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_CHUNK_DURATION = 0.2
DEFAULT_ALGORITHM_MODE = "CREPE Only"

# Algorithm Options
ALGORITHM_MODES = [
    "Multi-Algorithm (Best)",
    "CREPE Only",
    "Yin Only", 
    "YinFFT Only", 
    "MComb Only", 
    "Schmitt Only"
]

# File Paths
GROUND_TRUTH_PATH = "ground_truth.json"
SCALE_TRAINING_PATH = "scales/c_major_scale.json"

# Scale Training Configuration
DEFAULT_TEMPO_BPM = 60
TEMPO_MIN_BPM = 30
TEMPO_MAX_BPM = 180
NOTE_TIMING_TOLERANCE = 0.2  # seconds tolerance for note timing

# UI Configuration
PAGE_TITLE = "🎸 Agentic Music Tutor"
PAGE_ICON = "🎸"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
