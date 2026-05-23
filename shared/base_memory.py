"""
Base Memory Interface
Abstract base class for conversation memory implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime

class BaseConversationMemory(ABC):
    """Abstract base class for conversation memory"""

    @abstractmethod
    def add_interaction(self, user_input: str, assistant_response: str):
        """Add a user-assistant interaction to memory"""
        pass

    @abstractmethod
    def get_context(self) -> List[Dict[str, str]]:
        """Get the current conversation context"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all conversation history"""
        pass

    @abstractmethod
    def export_conversation(self) -> str:
        """Export conversation history"""
        pass

class MemoryManager:
    """Utility class for managing different memory implementations"""

    @staticmethod
    def create_memory(memory_type: str = "buffer", **kwargs) -> BaseConversationMemory:
        """
        Factory method for creating memory instances

        Args:
            memory_type: Type of memory to create
            **kwargs: Additional arguments

        Returns:
            Memory instance
        """
        if memory_type == "buffer":
            from oss_assistant.memory import ConversationMemory
            return ConversationMemory(**kwargs)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    @staticmethod
    def compare_memories(memory1: BaseConversationMemory, memory2: BaseConversationMemory) -> Dict[str, any]:
        """
        Compare two memory instances

        Args:
            memory1: First memory instance
            memory2: Second memory instance

        Returns:
            Comparison statistics
        """
        history1 = memory1.get_context()
        history2 = memory2.get_context()

        return {
            "memory1_turns": len(history1),
            "memory2_turns": len(history2),
            "turn_difference": abs(len(history1) - len(history2)),
            "identical_history": history1 == history2
        }