"""Visualization module for creating charts and plots."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List


def create_waveform_plot(audio_data: np.ndarray, sample_rate: int, title: str = "Audio Waveform"):
    """Create a waveform plot using plotly."""
    if len(audio_data) == 0:
        return go.Figure()
    
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Audio',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_note_display_plot(detected_notes: List[str], expected_notes: List[str]):
    """Create a note comparison visualization."""
    if not detected_notes:
        return go.Figure()
    
    # Create subplot for detected vs expected
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Detected Notes', 'Expected Notes'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Detected notes
    if detected_notes:
        x_detected = list(range(len(detected_notes)))
        colors = ['#2E8B57' if note in expected_notes else '#DC143C' for note in detected_notes]
        
        fig.add_trace(
            go.Scatter(
                x=x_detected,
                y=detected_notes,
                mode='markers+text',
                marker=dict(size=20, color=colors),
                text=detected_notes,
                textposition="middle center",
                name='Detected',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Expected notes
    x_expected = list(range(len(expected_notes)))
    fig.add_trace(
        go.Scatter(
            x=x_expected,
            y=expected_notes,
            mode='markers+text',
            marker=dict(size=20, color='#4169E1'),
            text=expected_notes,
            textposition="middle center",
            name='Expected',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
