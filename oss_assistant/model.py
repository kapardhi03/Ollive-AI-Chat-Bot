"""
OSS Model Handler
Manages loading and inference for open-source models
"""

import logging
import os
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OSSModel:
    """Handler for open-source language models"""

    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load the specified model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name} on device: {self.device}")

            # Load tokenizer with proper padding token
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate precision and device mapping
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda" and torch.cuda.is_available():
                load_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            else:
                load_kwargs.update({
                    "torch_dtype": torch.float32
                })

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Move to CPU if CUDA not available
            if self.device == "cpu":
                self.model = self.model.to("cpu")

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                return_full_text=False  # Only return generated text, not prompt
            )

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a smaller model if the requested one fails
            if "Phi-3" not in self.model_name:
                logger.info("Attempting to load fallback model: microsoft/DialoGPT-medium")
                self.model_name = "microsoft/DialoGPT-medium"
                try:
                    self.load_model()
                except:
                    raise Exception(f"Failed to load both primary and fallback models: {str(e)}")
            else:
                raise

    def generate_response(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> str:
        """
        Generate a response to user input

        Args:
            user_input: The user's message
            conversation_context: Previous conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """
        try:
            if not self.pipeline:
                return "Model not loaded properly. Please check the logs."

            # Format the prompt with context
            formatted_prompt = self._format_prompt(user_input, conversation_context)

            # Generate response with proper parameters
            generation_kwargs = {
                "max_new_tokens": min(max_tokens, 500),  # Cap max tokens
                "temperature": max(0.1, min(temperature, 1.0)),  # Clamp temperature
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3
            }

            # Generate response
            outputs = self.pipeline(formatted_prompt, **generation_kwargs)

            # Extract the generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
            else:
                generated_text = ""

            # Clean up the response
            response = self._clean_response(generated_text)

            if not response or len(response.strip()) < 5:
                return "I apologize, but I couldn't generate a proper response. Please try again."

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def _clean_response(self, response: str) -> str:
        """Clean and validate the generated response"""
        if not response:
            return ""

        # Remove common artifacts
        cleaned = response.strip()

        # Remove repeated patterns
        lines = cleaned.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and (not unique_lines or line != unique_lines[-1]):
                unique_lines.append(line)

        cleaned = '\n'.join(unique_lines[:5])  # Limit to 5 lines max

        # Remove incomplete sentences at the end
        sentences = cleaned.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            cleaned = '.'.join(sentences[:-1]) + '.'

        return cleaned.strip()

    def _format_prompt(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format the prompt with conversation context using model-specific templates

        Args:
            user_input: Current user input
            conversation_context: Previous conversation history

        Returns:
            Formatted prompt string
        """
        # Use model-specific chat templates
        if "Phi-3" in self.model_name:
            return self._format_phi3_prompt(user_input, conversation_context)
        elif "Qwen" in self.model_name:
            return self._format_qwen_prompt(user_input, conversation_context)
        elif "Llama" in self.model_name:
            return self._format_llama_prompt(user_input, conversation_context)
        else:
            return self._format_generic_prompt(user_input, conversation_context)

    def _format_phi3_prompt(self, user_input: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt for Phi-3 models"""
        system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."

        prompt_parts = [f"<|system|>\n{system_prompt}<|end|>"]

        # Add conversation context
        if conversation_context:
            for turn in conversation_context[-3:]:  # Keep last 3 turns for context
                prompt_parts.append(f"<|user|>\n{turn['user']}<|end|>")
                prompt_parts.append(f"<|assistant|>\n{turn['assistant']}<|end|>")

        # Add current user input
        prompt_parts.append(f"<|user|>\n{user_input}<|end|>")
        prompt_parts.append("<|assistant|>")

        return "\n".join(prompt_parts)

    def _format_qwen_prompt(self, user_input: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt for Qwen models"""
        system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."

        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]

        # Add conversation context
        if conversation_context:
            for turn in conversation_context[-3:]:
                prompt_parts.append(f"<|im_start|>user\n{turn['user']}<|im_end|>")
                prompt_parts.append(f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>")

        # Add current user input
        prompt_parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant")

        return "\n".join(prompt_parts)

    def _format_llama_prompt(self, user_input: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt for Llama models"""
        system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."

        prompt_parts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"]

        # Add conversation context
        if conversation_context:
            for turn in conversation_context[-3:]:
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{turn['user']}<|eot_id|>")
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{turn['assistant']}<|eot_id|>")

        # Add current user input
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")

        return "\n".join(prompt_parts)

    def _format_generic_prompt(self, user_input: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt for generic models"""
        system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."

        prompt_parts = [f"System: {system_prompt}"]

        # Add conversation context
        if conversation_context:
            for turn in conversation_context[-3:]:
                prompt_parts.append(f"Human: {turn['user']}")
                prompt_parts.append(f"Assistant: {turn['assistant']}")

        # Add current user input
        prompt_parts.append(f"Human: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_size": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters"
        }