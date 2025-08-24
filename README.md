# Agentic_Music_Tutor

An **autonomous AI-powered music learning application** that provides real-time feedback on scale practice with intelligent analysis, personalized insights, and **LangChain-powered agentic tutoring**.

## Features

### Core Functionality
- **Real-time Note Detection** - Detects musical notes from microphone input using multiple algorithms
- **Scale Training Mode** - Guided practice for major/minor scales with tempo control
- **Free Play Mode** - Unrestricted note detection practice
- **Timing Analysis** - Measures accuracy of note timing during scale practice

### AI-Powered Analysis
- **LLM Integration** - Uses Ollama for intelligent performance analysis
- **Enhanced Audio Analysis** - Analyzes waveform quality, sustain, attack, and noise
- **Comprehensive Feedback** - Provides specific, actionable insights on timing and technique
- **Performance Rating** - Overall assessment combining timing and audio quality

### **NEW: Agentic AI Tutor (LangChain)**
- **Autonomous Learning Management** - AI that guides your entire musical journey
- **Adaptive Difficulty** - Automatically adjusts based on your skill level and progress
- **Natural Language Interaction** - Chat with your AI tutor using natural language
- **Intelligent Practice Planning** - AI creates personalized practice sessions and goals
- **Learning Path Orchestration** - Manages warmup, main practice, assessment, and cooldown phases
- **Student Profile Management** - Tracks your progress, strengths, and areas for improvement

### Available Scales
- C Major, G Major, D Major
- A Minor, E Minor
- Each with configurable tempo (30-180 BPM)

## Installation

### Prerequisites
1. Python 3.8+
2. Ollama (for LLM analysis and agentic tutoring)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Agentic_Music_Tutor

# Install dependencies (including LangChain)
pip install -r requirements.txt

# Install and start Ollama (for AI analysis and agentic tutoring)
# Visit: https://ollama.ai/
ollama pull llama3.2:3b
ollama serve
```

## Usage

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
4. Click "Get AI Analysis" for personalized feedback

### **Using the Agentic AI Tutor**
1. Select "AI Agentic Tutor" mode
2. **Chat with AI**: Start a natural language conversation about your musical goals
3. **Autonomous Sessions**: Let AI create personalized practice sessions
4. **Adaptive Learning**: AI automatically adjusts difficulty and focus areas
5. **Progress Tracking**: Monitor your learning journey with AI insights

**Example AI Tutor Interactions:**
- "I want to improve my timing on scales"
- "I'm feeling stuck on the C Major scale"
- "What should I practice next?"
- "Create a practice plan for me"
- "How am I progressing?"

## Testing

Run the integration tests to verify functionality:

### Test LLM Integration
```bash
python test_llm_integration.py
```

### Test LangChain Integration
```bash
python test_langchain_integration.py
```

### Test All Components
```bash
# Test both integrations
python test_llm_integration.py && python test_langchain_integration.py
```

## Technical Details

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

### **LangChain Agentic Integration**
- **Autonomous Decision Making**: AI decides what to teach next
- **Adaptive Learning Paths**: Personalizes based on student performance
- **Dynamic Goal Setting**: Sets and adjusts practice objectives
- **Conversational Interface**: Natural language interaction
- **Proactive Guidance**: Suggests improvements before problems occur
- **Learning Orchestration**: Manages complete practice sessions
- **Student Profiling**: Tracks progress and adapts instruction

## Future Enhancements

- Support for more instruments
- Advanced rhythm training
- Custom scale creation
- Performance history tracking
- Integration with music theory lessons
