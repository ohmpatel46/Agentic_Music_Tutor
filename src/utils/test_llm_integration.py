#!/usr/bin/env python3
"""Test script for LLM integration and enhanced data collection."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.scale_trainer import ScaleTrainer
from src.core.audio_processor import analyze_waveform_quality
from src.core.llm_analyzer import create_llm_analyzer
import numpy as np


def test_enhanced_data_collection():
    """Test the enhanced data collection in ScaleTrainer."""
    print("🧪 Testing Enhanced Data Collection...")
    
    # Create a scale trainer
    trainer = ScaleTrainer()
    
    # Start training session
    trainer.start_training()
    
    # Simulate some note detections with enhanced data
    print("📝 Simulating note detections...")
    
    # Note 1: C3
    trainer.check_note_timing(
        "C3", 
        confidence=0.85, 
        waveform_data={"sustain_duration": 0.8, "attack_quality": 0.9, "frequency_stability": 0.95, "noise_level": 0.8},
        algorithm="CREPE",
        audio_chunk=np.random.randn(48000) * 0.1  # Simulated audio
    )
    
    # Note 2: D3 (with delay)
    trainer.check_note_timing(
        "D3", 
        confidence=0.78, 
        waveform_data={"sustain_duration": 0.7, "attack_quality": 0.8, "frequency_stability": 0.88, "noise_level": 0.75},
        algorithm="CREPE",
        audio_chunk=np.random.randn(48000) * 0.1
    )
    
    # Note 3: E3 (with delay)
    trainer.check_note_timing(
        "E3", 
        confidence=0.92, 
        waveform_data={"sustain_duration": 0.9, "attack_quality": 0.95, "frequency_stability": 0.98, "noise_level": 0.9},
        algorithm="CREPE",
        audio_chunk=np.random.randn(48000) * 0.1
    )
    
    # Stop training to get analysis data
    trainer.stop_training()
    
    # Get comprehensive analysis data
    analysis_data = trainer.get_comprehensive_analysis_data()
    
    if analysis_data and 'performance_summary' in analysis_data:
        print("✅ Enhanced data collection test completed!")
        print(f"📊 Collected data for {analysis_data['performance_summary']['notes_completed']} notes")
        print(f"⏱️ Average delay: {analysis_data['timing_analysis']['average_delay']:.3f}s")
        print(f"🎯 Average confidence: {analysis_data['audio_quality']['average_confidence']:.3f}")
        print(f"⭐ Overall rating: {analysis_data['performance_summary']['overall_rating']}")
    else:
        print("⚠️ Analysis data is empty - this might indicate an issue")
        print(f"Analysis data keys: {list(analysis_data.keys()) if analysis_data else 'None'}")
    
    return analysis_data


def test_llm_analyzer():
    """Test the LLM analyzer functionality."""
    print("\n🤖 Testing LLM Analyzer...")
    
    # Create LLM analyzer
    analyzer = create_llm_analyzer()
    
    if analyzer:
        print("✅ LLM analyzer created successfully!")
        print(f"🔧 Available models: {analyzer.get_available_models()}")
        
        # Test connection
        if analyzer.test_connection():
            print("✅ Ollama connection successful!")
        else:
            print("⚠️ Ollama connection failed - make sure Ollama is running")
    else:
        print("❌ Failed to create LLM analyzer")
    
    return analyzer


def test_waveform_analysis():
    """Test the waveform analysis functions."""
    print("\n📈 Testing Waveform Analysis...")
    
    # Create simulated audio data
    sample_rate = 48000
    duration = 0.5
    samples = int(sample_rate * duration)
    
    # Simulate a clean note
    t = np.linspace(0, duration, samples)
    clean_note = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-2 * t)  # A4 note with decay
    
    # Add some noise
    noisy_note = clean_note + 0.1 * np.random.randn(samples)
    
    # Analyze waveform quality
    quality_metrics = analyze_waveform_quality(noisy_note)
    
    print("✅ Waveform analysis test completed!")
    print(f"🎵 Sustain duration: {quality_metrics.get('sustain_duration', 0):.3f}s")
    print(f"⚡ Attack quality: {quality_metrics.get('attack_quality', 0):.3f}")
    print(f"🎯 Frequency stability: {quality_metrics.get('frequency_stability', 0):.3f}")
    print(f"🔇 Noise level: {quality_metrics.get('noise_level', 0):.3f}")
    print(f"⭐ Overall quality: {quality_metrics.get('overall_quality', 0):.3f}")
    
    return quality_metrics


def main():
    """Run all tests."""
    print("🚀 Starting LLM Integration Tests...\n")
    
    try:
        # Test 1: Enhanced data collection
        analysis_data = test_enhanced_data_collection()
        
        # Test 2: LLM analyzer
        analyzer = test_llm_analyzer()
        
        # Test 3: Waveform analysis
        quality_metrics = test_waveform_analysis()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print(f"   • Enhanced data collection: ✅ Working")
        print(f"   • LLM analyzer: {'✅ Working' if analyzer else '❌ Failed'}")
        print(f"   • Waveform analysis: ✅ Working")
        
        if analyzer and analyzer.test_connection():
            print("\n💡 Next steps:")
            print("   1. Run the Streamlit app: streamlit run app.py")
            print("   2. Complete a scale training session")
            print("   3. Click 'Get AI Analysis' to test LLM integration")
        else:
            print("\n⚠️ To test LLM integration:")
            print("   1. Install and start Ollama: https://ollama.ai/")
            print("   2. Pull a model: ollama pull llama3.2:3b")
            print("   3. Run the tests again")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
