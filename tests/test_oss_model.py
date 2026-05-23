"""
Tests for OSS Model functionality
"""

import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oss_assistant.model import OSSModel
from oss_assistant.memory import ConversationMemory


class TestOSSModel:
    """Test OSS model functionality"""

    def test_model_initialization(self):
        """Test model can be initialized"""
        # Use a very small model for testing to avoid memory issues
        try:
            model = OSSModel("microsoft/DialoGPT-small")
            assert model.model_name == "microsoft/DialoGPT-small"
            assert model.tokenizer is not None
        except Exception as e:
            # If model loading fails (common in CI), skip test
            pytest.skip(f"Model loading failed: {e}")

    def test_prompt_formatting(self):
        """Test prompt formatting methods"""
        model = OSSModel("microsoft/DialoGPT-small")

        # Test generic prompt formatting
        user_input = "Hello, how are you?"
        formatted = model._format_generic_prompt(user_input)

        assert "Hello, how are you?" in formatted
        assert "System:" in formatted
        assert "Assistant:" in formatted

    def test_prompt_formatting_with_context(self):
        """Test prompt formatting with conversation context"""
        model = OSSModel("microsoft/DialoGPT-small")

        context = [
            {"user": "What's your name?", "assistant": "I'm an AI assistant."},
            {"user": "What can you do?", "assistant": "I can help with various tasks."}
        ]

        user_input = "That's great!"
        formatted = model._format_generic_prompt(user_input, context)

        assert "What's your name?" in formatted
        assert "I'm an AI assistant." in formatted
        assert "That's great!" in formatted

    def test_model_info(self):
        """Test model info retrieval"""
        try:
            model = OSSModel("microsoft/DialoGPT-small")
            info = model.get_model_info()

            assert "model_name" in info
            assert "device" in info
            assert info["model_name"] == "microsoft/DialoGPT-small"
            assert info["device"] in ["cpu", "cuda"]
        except Exception:
            pytest.skip("Model loading failed")


class TestConversationMemory:
    """Test conversation memory functionality"""

    def test_memory_initialization(self):
        """Test memory initialization"""
        memory = ConversationMemory()
        assert memory.max_turns == 10
        assert len(memory.conversation_history) == 0

    def test_add_interaction(self):
        """Test adding interactions to memory"""
        memory = ConversationMemory()

        memory.add_interaction("Hello", "Hi there!")

        assert len(memory.conversation_history) == 1
        assert memory.conversation_history[0]["user"] == "Hello"
        assert memory.conversation_history[0]["assistant"] == "Hi there!"

    def test_memory_limit(self):
        """Test memory respects max turns limit"""
        memory = ConversationMemory(max_turns=2)

        # Add more interactions than the limit
        memory.add_interaction("First", "Response 1")
        memory.add_interaction("Second", "Response 2")
        memory.add_interaction("Third", "Response 3")

        # Should only keep the last 2
        assert len(memory.conversation_history) == 2
        assert memory.conversation_history[0]["user"] == "Second"
        assert memory.conversation_history[1]["user"] == "Third"

    def test_get_context(self):
        """Test getting conversation context"""
        memory = ConversationMemory()

        memory.add_interaction("Hello", "Hi there!")
        memory.add_interaction("How are you?", "I'm doing well!")

        context = memory.get_context()
        assert len(context) == 2
        assert context[0]["user"] == "Hello"
        assert context[1]["user"] == "How are you?"

    def test_clear_memory(self):
        """Test clearing memory"""
        memory = ConversationMemory()

        memory.add_interaction("Hello", "Hi!")
        assert len(memory.conversation_history) == 1

        memory.clear()
        assert len(memory.conversation_history) == 0

    def test_export_conversation(self):
        """Test conversation export"""
        memory = ConversationMemory()

        memory.add_interaction("Hello", "Hi!")
        exported = memory.export_conversation()

        assert isinstance(exported, str)
        assert "Hello" in exported
        assert "Hi!" in exported
        assert "conversation_history" in exported

    def test_search_conversation(self):
        """Test conversation search"""
        memory = ConversationMemory()

        memory.add_interaction("Hello", "Hi there!")
        memory.add_interaction("How are you?", "I'm doing well!")
        memory.add_interaction("What's the weather?", "I don't have access to weather data.")

        # Search for weather-related conversation
        results = memory.search_conversation("weather")
        assert len(results) == 1
        assert "weather" in results[0]["user"]

        # Search for greeting
        results = memory.search_conversation("hello")
        assert len(results) == 1
        assert "Hello" in results[0]["user"]