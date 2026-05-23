"""
Prompt Templates for OSS Assistant
Contains various prompt templates and formatting utilities
"""

from typing import List, Dict, Optional

class PromptTemplates:
    """Collection of prompt templates for different use cases"""

    # System prompts
    SYSTEM_PROMPTS = {
        "default": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
        "concise": "You are a helpful AI assistant. Provide brief, accurate answers.",
        "detailed": "You are a helpful AI assistant. Provide comprehensive and detailed responses with examples when helpful.",
        "safe": "You are a helpful AI assistant. Always prioritize safety and refuse harmful requests. Be honest about your limitations."
    }

    # Chat templates for different models
    CHAT_TEMPLATES = {
        "phi3": """<|system|>
{system_prompt}<|end|>
{conversation_history}
<|user|>
{user_input}<|end|>
<|assistant|>""",

        "qwen": """<|im_start|>system
{system_prompt}<|im_end|>
{conversation_history}
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant""",

        "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
{conversation_history}
<|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",

        "generic": """{system_prompt}

{conversation_history}
Human: {user_input}
Assistant:"""
    }

    @classmethod
    def format_chat_prompt(
        cls,
        user_input: str,
        model_type: str = "generic",
        system_prompt_type: str = "default",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a chat prompt using the appropriate template

        Args:
            user_input: The user's current input
            model_type: Type of model (phi3, qwen, llama, generic)
            system_prompt_type: Type of system prompt to use
            conversation_history: Previous conversation turns

        Returns:
            Formatted prompt string
        """
        # Get system prompt
        system_prompt = cls.SYSTEM_PROMPTS.get(system_prompt_type, cls.SYSTEM_PROMPTS["default"])

        # Format conversation history
        history_str = ""
        if conversation_history:
            history_str = cls._format_conversation_history(conversation_history, model_type)

        # Get template
        template = cls.CHAT_TEMPLATES.get(model_type, cls.CHAT_TEMPLATES["generic"])

        # Format the complete prompt
        return template.format(
            system_prompt=system_prompt,
            conversation_history=history_str,
            user_input=user_input
        )

    @classmethod
    def _format_conversation_history(
        cls,
        history: List[Dict[str, str]],
        model_type: str
    ) -> str:
        """Format conversation history based on model type"""
        if not history:
            return ""

        formatted_turns = []

        if model_type == "phi3":
            for turn in history:
                formatted_turns.append(f"<|user|>\n{turn['user']}<|end|>")
                formatted_turns.append(f"<|assistant|>\n{turn['assistant']}<|end|>")

        elif model_type == "qwen":
            for turn in history:
                formatted_turns.append(f"<|im_start|>user\n{turn['user']}<|im_end|>")
                formatted_turns.append(f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>")

        elif model_type == "llama":
            for turn in history:
                formatted_turns.append(f"<|start_header_id|>user<|end_header_id|>\n{turn['user']}<|eot_id|>")
                formatted_turns.append(f"<|start_header_id|>assistant<|end_header_id|>\n{turn['assistant']}<|eot_id|>")

        else:  # generic
            for turn in history:
                formatted_turns.append(f"Human: {turn['user']}")
                formatted_turns.append(f"Assistant: {turn['assistant']}")

        return "\n".join(formatted_turns)

    @classmethod
    def create_safety_prompt(cls, user_input: str) -> str:
        """
        Create a safety-focused prompt

        Args:
            user_input: User's input to evaluate

        Returns:
            Safety-focused prompt
        """
        return f"""You are a helpful AI assistant with strong safety guidelines.
Always refuse requests for:
- Harmful or illegal activities
- Personal information about individuals
- Generating malicious code
- Spreading misinformation

User request: {user_input}

Please provide a helpful and safe response:"""

    @classmethod
    def create_factual_prompt(cls, user_input: str) -> str:
        """
        Create a fact-focused prompt

        Args:
            user_input: User's input

        Returns:
            Fact-focused prompt
        """
        return f"""You are a knowledgeable AI assistant focused on providing accurate information.
If you're unsure about facts, clearly state your uncertainty.
Cite sources when possible and distinguish between facts and opinions.

Question: {user_input}

Please provide an accurate and well-sourced response:"""