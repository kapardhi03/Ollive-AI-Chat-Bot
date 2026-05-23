"""
Conversation Memory Management for Frontier Assistant
Identical to OSS assistant memory for consistent comparison
"""

from typing import List, Dict, Optional
from datetime import datetime
import json

class ConversationMemory:
    """Manages conversation history and context for the frontier assistant"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[Dict[str, str]] = []
        self.session_start = datetime.now()

    def add_interaction(self, user_input: str, assistant_response: str):
        """
        Add a user-assistant interaction to memory

        Args:
            user_input: The user's message
            assistant_response: The assistant's response
        """
        interaction = {
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        }

        self.conversation_history.append(interaction)

        # Keep only the most recent turns
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]

    def get_context(self) -> List[Dict[str, str]]:
        """
        Get the current conversation context

        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()

    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, str]]:
        """
        Get recent conversation context

        Args:
            num_turns: Number of recent turns to return

        Returns:
            List of recent conversation turns
        """
        return self.conversation_history[-num_turns:] if self.conversation_history else []

    def clear(self):
        """Clear all conversation history"""
        self.conversation_history = []
        self.session_start = datetime.now()

    def get_conversation_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current conversation

        Returns:
            Dictionary with conversation statistics
        """
        return {
            "total_turns": len(self.conversation_history),
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }

    def export_conversation(self) -> str:
        """
        Export conversation history as JSON

        Returns:
            JSON string of conversation history
        """
        export_data = {
            "session_start": self.session_start.isoformat(),
            "conversation_history": self.conversation_history,
            "summary": self.get_conversation_summary()
        }
        return json.dumps(export_data, indent=2)

    def search_conversation(self, query: str) -> List[Dict[str, str]]:
        """
        Search conversation history for relevant interactions

        Args:
            query: Search query

        Returns:
            List of matching interactions
        """
        query_lower = query.lower()
        matching_interactions = []

        for interaction in self.conversation_history:
            if (query_lower in interaction["user"].lower() or
                query_lower in interaction["assistant"].lower()):
                matching_interactions.append(interaction)

        return matching_interactions