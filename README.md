# Agentic_Music_Tutor

An AI-powered music learning application that provides real-time feedback on scale practice with intelligent analysis and personalized insights.

## 🎵 Features

### Core Functionality
- **Real-time Note Detection** - Detects musical notes from microphone input using multiple algorithms
- **Scale Training Mode** - Guided practice for major/minor scales with tempo control
- **Free Play Mode** - Unrestricted note detection practice
- **Timing Analysis** - Measures accuracy of note timing during scale practice

### 🚀 New: AI-Powered Analysis
- **LLM Integration** - Uses Ollama for intelligent performance analysis
- **Enhanced Audio Analysis** - Analyzes waveform quality, sustain, attack, and noise
- **Comprehensive Feedback** - Provides specific, actionable insights on timing and technique
- **Performance Rating** - Overall assessment combining timing and audio quality

### 🎼 Available Scales
- C Major, G Major, D Major
- A Minor, E Minor
- Each with configurable tempo (30-180 BPM)

## 🔧 Installation

### Prerequisites
1. Python 3.8+
2. Ollama (for LLM analysis)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Agentic_Music_Tutor

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (for AI analysis)
# Visit: https://ollama.ai/
ollama pull llama3.2:3b
```

## 🚀 Usage

### Basic Usage
```bash
# Start the application
streamlit run app.py
```

### LLM Analysis Setup
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
2. **Start Ollama**: Run `ollama serve` in terminal
3. **Pull Model**: `ollama pull llama3.2:3b`
4. **Run App**: `streamlit run app.py`

### Using AI Analysis
1. Select "Scale Training" mode
2. Choose a scale and tempo
3. Complete a training session
4. Click "🧠 Get AI Analysis" for personalized feedback

## 🧪 Testing

Run the integration tests to verify functionality:
```bash
python test_llm_integration.py
```

## 📊 Technical Details

### Audio Processing Pipeline
1. **Microphone Input** → 0.2s audio chunks (48kHz, stereo→mono)
2. **Note Detection** → Multiple algorithms (CREPE, aubio-based)
3. **Waveform Analysis** → Sustain, attack, frequency stability, noise level
4. **Timing Analysis** → Note-to-note delays and consistency
5. **LLM Analysis** → AI-powered insights and recommendations

### Data Collection
- **Timing Data**: Note delays, consistency scores
- **Audio Quality**: Confidence scores, waveform metrics
- **Algorithm Performance**: Detection method tracking
- **Performance Metrics**: Overall rating, session statistics

### LLM Integration
- **Ollama API**: Local LLM processing
- **Comprehensive Prompts**: Detailed performance analysis
- **Actionable Feedback**: Specific improvement suggestions
- **Musical Terminology**: Expert-level insights

## 🎯 Future Enhancements

- [ ] Support for more instruments
- [ ] Advanced rhythm training
- [ ] Custom scale creation
- [ ] Performance history tracking
- [ ] Integration with music theory lessons

## 📝 Notes
- Make frontend interface
- Enable auto spec detect through code for mic
- Customizable threshold buttons and other audio detection controls from settings tab
- Enhanced with AI-powered analysis and comprehensive audio quality assessment