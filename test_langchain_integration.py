"""Test script for LangChain integration in the Agentic Music Tutor."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.music_agent import create_music_agent, PracticeFocus, LearningStage
from src.core.conversation_manager import create_conversation_manager
from src.core.practice_orchestrator import create_practice_orchestrator


def test_music_agent():
    """Test the music agent creation and basic functionality."""
    print("🧪 Testing Music Agent...")
    
    try:
        # Test agent creation
        agent = create_music_agent()
        if agent:
            print("✅ Music Agent created successfully")
            
            # Test student profile
            profile = agent.student_profile
            print(f"   - Student Level: {profile.level.value}")
            print(f"   - Instrument: {profile.preferred_instrument}")
            print(f"   - Practice Time: {profile.practice_time_per_day} minutes/day")
            
            # Test basic guidance
            guidance = agent.get_agent_guidance("Hello, I'm new to music!")
            print(f"   - Basic Guidance: {guidance[:100]}...")
            
            return True
        else:
            print("❌ Failed to create Music Agent")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Music Agent: {e}")
        return False


def test_conversation_manager():
    """Test the conversation manager."""
    print("\n🧪 Testing Conversation Manager...")
    
    try:
        # Create agent first
        agent = create_music_agent()
        if not agent:
            print("❌ Skipping Conversation Manager test - no Music Agent")
            return False
        
        # Test conversation manager creation
        conv_manager = create_conversation_manager(agent)
        if conv_manager:
            print("✅ Conversation Manager created successfully")
            
            # Test conversation start
            response = conv_manager.start_conversation("Hi, I want to learn guitar!")
            print(f"   - Conversation Response: {response[:100]}...")
            
            return True
        else:
            print("❌ Failed to create Conversation Manager")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Conversation Manager: {e}")
        return False


def test_practice_orchestrator():
    """Test the practice orchestrator."""
    print("\n🧪 Testing Practice Orchestrator...")
    
    try:
        # Create dependencies first
        agent = create_music_agent()
        if not agent:
            print("❌ Skipping Practice Orchestrator test - no Music Agent")
            return False
        
        conv_manager = create_conversation_manager(agent)
        if not conv_manager:
            print("❌ Skipping Practice Orchestrator test - no Conversation Manager")
            return False
        
        # Test orchestrator creation
        orchestrator = create_practice_orchestrator(agent, conv_manager)
        if orchestrator:
            print("✅ Practice Orchestrator created successfully")
            
            # Test learning paths
            paths = orchestrator.learning_paths
            print(f"   - Learning Paths: {len(paths)} levels available")
            for level, config in paths.items():
                print(f"     * {level}: {len(config['scales'])} scales, {len(config['phases'])} phases")
            
            return True
        else:
            print("❌ Failed to create Practice Orchestrator")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Practice Orchestrator: {e}")
        return False


def test_integration():
    """Test the complete integration."""
    print("\n🧪 Testing Complete Integration...")
    
    try:
        # Test all components
        agent_ok = test_music_agent()
        conv_ok = test_conversation_manager()
        orch_ok = test_practice_orchestrator()
        
        if agent_ok and conv_ok and orch_ok:
            print("\n🎉 All tests passed! LangChain integration is working.")
            return True
        else:
            print("\n❌ Some tests failed. Check the output above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Integration test error: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Testing LangChain Integration for Agentic Music Tutor")
    print("=" * 60)
    
    # Check if Ollama is available
    print("🔍 Checking Ollama availability...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models available")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.get('name', 'Unknown')}")
        else:
            print("⚠️ Ollama responded but with error status")
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print("💡 Install a model: ollama pull llama3.2:3b")
        return False
    
    # Run tests
    success = test_integration()
    
    if success:
        print("\n🎯 Ready to use the Agentic Music Tutor!")
        print("💡 Run: streamlit run app.py")
    else:
        print("\n🔧 Some components need attention. Check the errors above.")
    
    return success


if __name__ == "__main__":
    main()
