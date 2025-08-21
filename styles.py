"""CSS styles for the Agentic Music Tutor application."""

MAIN_STYLES = """
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.note-display {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    margin: 1rem 0;
}

.stats-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}

.recording-status {
    padding: 0.5rem 1rem;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}

.recording-active {
    background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
}

.recording-ready {
    background: linear-gradient(90deg, #74b9ff 0%, #0984e3 100%);
    color: white;
}

.control-button {
    margin: 0.25rem 0;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

.debug-info {
    background: #f8f9fa;
    padding: 0.5rem;
    border-radius: 5px;
    border-left: 3px solid #6c757d;
    font-family: monospace;
    font-size: 0.9rem;
}
</style>
"""

def get_main_header_html():
    """Get the main header HTML with styling."""
    return """
    <div class="main-header">
        <h1>🎸 Agentic Music Tutor</h1>
        <p>Real-time note detection and feedback</p>
    </div>
    """

def get_note_display_html(notes_text: str):
    """Get the note display HTML with styling."""
    return f'<div class="note-display">{notes_text}</div>'
