"""
API Client for Frontier Models
Handles communication with various frontier model APIs (OpenAI, Anthropic, Google, etc.)
"""

import logging
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
import openai
from anthropic import Anthropic
try:
    # Try the new package first
    import google.genai as genai
    GOOGLE_GENAI_NEW = True
except ImportError:
    try:
        # Fallback to the deprecated package
        import google.generativeai as genai
        GOOGLE_GENAI_NEW = False
    except ImportError:
        # No Google package available
        genai = None
        GOOGLE_GENAI_NEW = None
import requests
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrontierModelClient:
    """Client for communicating with various frontier model APIs"""

    def __init__(self):
        self.api_keys = {}
        self.clients = {}
        self.usage_stats = {
            "requests": 0,
            "tokens_used": 0,
            "total_cost": 0.0,
            "last_request": None
        }
        self.rate_limits = {
            "openai": {"requests_per_minute": 60, "last_request": 0, "request_count": 0},
            "anthropic": {"requests_per_minute": 60, "last_request": 0, "request_count": 0},
            "google": {"requests_per_minute": 60, "last_request": 0, "request_count": 0},
            "deepseek": {"requests_per_minute": 60, "last_request": 0, "request_count": 0}
        }
        self._lock = Lock()

    def set_api_key(self, model: str, api_key: str):
        """Set API key for a specific model provider"""
        provider = self._get_provider_from_model(model)
        self.api_keys[provider] = api_key
        self._initialize_client(model, api_key)

    def _initialize_client(self, model: str, api_key: str):
        """Initialize the appropriate client based on the model"""
        try:
            provider = self._get_provider_from_model(model)

            if provider == "openai":
                self.clients["openai"] = openai.OpenAI(api_key=api_key)
            elif provider == "anthropic":
                self.clients["anthropic"] = Anthropic(api_key=api_key)
            elif provider == "google":
                if genai:
                    genai.configure(api_key=api_key)
                    # Use confirmed working model as default
                    if "3.1" in model and "flash-lite" in model:
                        model_name = model  # Use exact name for confirmed working model
                    elif "flash" in model or "lite" in model:
                        model_name = "gemini-3.1-flash-lite-preview"  # Use confirmed working model
                    elif "pro" in model:
                        model_name = "gemini-3.1-pro-preview"  # Try pro version
                    else:
                        model_name = "gemini-3.1-flash-lite-preview"  # Safe default to confirmed working
                    self.clients["google"] = genai.GenerativeModel(model_name)
                else:
                    raise ImportError("google-generativeai package not installed")
            elif provider == "deepseek":
                self.clients["deepseek"] = {
                    "api_key": api_key,
                    "base_url": "https://api.deepseek.com/v1"
                }

            logger.info(f"Initialized {provider} client for {model}")

        except Exception as e:
            logger.error(f"Error initializing client for {model}: {str(e)}")
            raise

    def _get_provider_from_model(self, model: str) -> str:
        """Get provider name from model identifier"""
        if model.startswith("gpt-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif model.startswith("gemini-"):
            return "google"
        elif model.startswith("deepseek-"):
            return "deepseek"
        else:
            raise ValueError(f"Unknown provider for model: {model}")

    def _check_rate_limit(self, provider: str):
        """Check and enforce rate limits"""
        with self._lock:
            current_time = time.time()
            rate_limit = self.rate_limits[provider]

            # Reset counter if more than a minute has passed
            if current_time - rate_limit["last_request"] > 60:
                rate_limit["request_count"] = 0

            # Check if we're within rate limits
            if rate_limit["request_count"] >= rate_limit["requests_per_minute"]:
                wait_time = 60 - (current_time - rate_limit["last_request"])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached for {provider}. Waiting {wait_time:.1f} seconds.")
                    time.sleep(wait_time)
                    rate_limit["request_count"] = 0

            # Update counters
            rate_limit["request_count"] += 1
            rate_limit["last_request"] = current_time

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for API call"""
        # Pricing per 1K tokens (approximate rates as of 2024)
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "deepseek-chat": {"input": 0.0014, "output": 0.0028}
        }

        model_pricing = pricing.get(model, {"input": 0.002, "output": 0.006})
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    def generate_response(
        self,
        model: str,
        user_input: str,
        conversation_context: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt_type: str = "default"
    ) -> str:
        """
        Generate a response using the specified frontier model

        Args:
            model: Model identifier
            user_input: User's current input
            conversation_context: Previous conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt_type: Type of system prompt to use

        Returns:
            Generated response string
        """
        try:
            provider = self._get_provider_from_model(model)

            # Check rate limits
            self._check_rate_limit(provider)

            # Generate response based on provider
            start_time = time.time()
            if provider == "openai":
                response = self._generate_openai_response(
                    model, user_input, conversation_context, temperature, max_tokens, system_prompt_type
                )
            elif provider == "anthropic":
                response = self._generate_anthropic_response(
                    model, user_input, conversation_context, temperature, max_tokens, system_prompt_type
                )
            elif provider == "google":
                response = self._generate_google_response(
                    model, user_input, conversation_context, temperature, max_tokens, system_prompt_type
                )
            elif provider == "deepseek":
                response = self._generate_deepseek_response(
                    model, user_input, conversation_context, temperature, max_tokens, system_prompt_type
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Update usage stats
            end_time = time.time()
            self._update_usage_stats(model, user_input, response, end_time - start_time)

            return response

        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def _update_usage_stats(self, model: str, user_input: str, response: str, latency: float):
        """Update usage statistics"""
        with self._lock:
            # Rough token estimation (1 token ≈ 4 characters)
            input_tokens = len(user_input) // 4
            output_tokens = len(response) // 4

            cost = self._calculate_cost(model, input_tokens, output_tokens)

            self.usage_stats["requests"] += 1
            self.usage_stats["tokens_used"] += input_tokens + output_tokens
            self.usage_stats["total_cost"] += cost
            self.usage_stats["last_request"] = datetime.now().isoformat()

            logger.info(f"Request completed. Tokens: {input_tokens + output_tokens}, "
                       f"Cost: ${cost:.4f}, Latency: {latency:.2f}s")

    def _generate_openai_response(
        self, model: str, user_input: str, conversation_context: List[Dict[str, str]],
        temperature: float, max_tokens: int, system_prompt_type: str
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            from .prompts import PromptTemplates
        except ImportError:
            from frontier_assistant.prompts import PromptTemplates

        client = self.clients.get("openai")
        if not client:
            if "openai" not in self.api_keys:
                raise ValueError("OpenAI API key not set. Please call set_api_key() with your OpenAI API key.")
            else:
                raise ValueError("OpenAI client not initialized. Please check your API key.")

        try:
            # Build messages
            messages = [
                {"role": "system", "content": PromptTemplates.SYSTEM_PROMPTS[system_prompt_type]}
            ]

            # Add conversation history
            if conversation_context:
                for turn in conversation_context[-5:]:  # Keep last 5 turns
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            # Add current user input
            messages.append({"role": "user", "content": user_input})

            # Make API call with retry logic
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=max(0.0, min(1.0, temperature)),
                        max_tokens=min(max_tokens, 4000),
                        timeout=30
                    )

                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip()
                    else:
                        raise ValueError("Empty response from OpenAI API")

                except openai.RateLimitError as e:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    raise e
                except openai.APITimeoutError as e:
                    if attempt < 2:
                        logger.warning(f"Timeout, retrying attempt {attempt + 2}/3...")
                        continue
                    raise e

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")

    def _generate_anthropic_response(
        self, model: str, user_input: str, conversation_context: List[Dict[str, str]],
        temperature: float, max_tokens: int, system_prompt_type: str
    ) -> str:
        """Generate response using Anthropic API"""
        try:
            from .prompts import PromptTemplates
        except ImportError:
            from frontier_assistant.prompts import PromptTemplates

        client = self.clients.get("anthropic")
        if not client:
            raise ValueError("Anthropic client not initialized. Please check your API key.")

        try:
            # Build messages
            messages = []

            # Add conversation history
            if conversation_context:
                for turn in conversation_context[-5:]:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            # Add current user input
            messages.append({"role": "user", "content": user_input})

            # Make API call with retry logic
            for attempt in range(3):
                try:
                    response = client.messages.create(
                        model=model,
                        system=PromptTemplates.SYSTEM_PROMPTS[system_prompt_type],
                        messages=messages,
                        temperature=max(0.0, min(1.0, temperature)),
                        max_tokens=min(max_tokens, 4000)
                    )

                    if response.content and len(response.content) > 0:
                        return response.content[0].text.strip()
                    else:
                        raise ValueError("Empty response from Anthropic API")

                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    elif attempt < 2:
                        logger.warning(f"Error occurred, retrying attempt {attempt + 2}/3...")
                        time.sleep(1)
                        continue
                    else:
                        raise e

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise Exception(f"Anthropic API error: {str(e)}")

    def _generate_google_response(
        self, model: str, user_input: str, conversation_context: List[Dict[str, str]],
        temperature: float, max_tokens: int, system_prompt_type: str
    ) -> str:
        """Generate response using Google Gemini API"""
        try:
            from .prompts import PromptTemplates
        except ImportError:
            from frontier_assistant.prompts import PromptTemplates

        client = self.clients.get("google")
        if not client:
            if "google" not in self.api_keys:
                raise ValueError("Google API key not set. Please call set_api_key() with your Google API key.")
            else:
                raise ValueError("Google client not initialized. Please check your API key.")

        try:
            # Build conversation history
            conversation_parts = [PromptTemplates.SYSTEM_PROMPTS[system_prompt_type]]

            if conversation_context:
                for turn in conversation_context[-3:]:  # Limit context for Gemini
                    conversation_parts.append(f"Human: {turn['user']}")
                    conversation_parts.append(f"Assistant: {turn['assistant']}")

            conversation_parts.append(f"Human: {user_input}")
            conversation_parts.append("Assistant:")

            full_prompt = "\n".join(conversation_parts)

            # Make API call with retry logic
            for attempt in range(3):
                try:
                    response = client.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=max(0.0, min(2.0, temperature)),  # Gemini supports up to 2.0
                            max_output_tokens=min(max_tokens, 8192)
                        )
                    )

                    if response.text and response.text.strip():
                        return response.text.strip()
                    else:
                        raise ValueError("Empty response from Google API")

                except Exception as e:
                    error_str = str(e).lower()
                    if "404" in error_str and "not found" in error_str:
                        # Model not found - try fallback model
                        if attempt == 0:
                            logger.warning(f"Model not found, trying fallback model...")
                            try:
                                client = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
                                response = client.generate_content(
                                    full_prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        temperature=max(0.0, min(2.0, temperature)),
                                        max_output_tokens=min(max_tokens, 8192)
                                    )
                                )
                                if response.text and response.text.strip():
                                    return response.text.strip()
                            except:
                                pass

                    if attempt < 2 and ("quota" in error_str or "rate" in error_str):
                        wait_time = 2 ** attempt
                        logger.warning(f"Google API rate limit, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    elif attempt < 2:
                        logger.warning(f"Error occurred, retrying attempt {attempt + 2}/3...")
                        time.sleep(1)
                        continue
                    else:
                        raise e

        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            # Return a fallback response for Google API issues
            return "I apologize, but I'm having trouble connecting to the Google Gemini API right now."

    def _generate_deepseek_response(
        self, model: str, user_input: str, conversation_context: List[Dict[str, str]],
        temperature: float, max_tokens: int, system_prompt_type: str
    ) -> str:
        """Generate response using DeepSeek API"""
        try:
            from .prompts import PromptTemplates
        except ImportError:
            from frontier_assistant.prompts import PromptTemplates

        client_config = self.clients.get("deepseek")
        if not client_config:
            raise ValueError("DeepSeek client not initialized. Please check your API key.")

        try:
            # Build messages (DeepSeek uses OpenAI-compatible format)
            messages = [
                {"role": "system", "content": PromptTemplates.SYSTEM_PROMPTS[system_prompt_type]}
            ]

            # Add conversation history
            if conversation_context:
                for turn in conversation_context[-5:]:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            # Add current user input
            messages.append({"role": "user", "content": user_input})

            headers = {
                "Authorization": f"Bearer {client_config['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": messages,
                "temperature": max(0.0, min(2.0, temperature)),
                "max_tokens": min(max_tokens, 4000),
                "stream": False
            }

            # Make API call with retry logic
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{client_config['base_url']}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("choices") and response_data["choices"][0].get("message", {}).get("content"):
                            return response_data["choices"][0]["message"]["content"].strip()
                        else:
                            raise ValueError("Empty response from DeepSeek API")
                    elif response.status_code == 429:  # Rate limit
                        if attempt < 2:
                            wait_time = 2 ** attempt
                            logger.warning(f"DeepSeek rate limit, retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"DeepSeek API rate limit: {response.text}")
                    else:
                        raise Exception(f"DeepSeek API error {response.status_code}: {response.text}")

                except requests.RequestException as e:
                    if attempt < 2:
                        logger.warning(f"Request failed, retrying attempt {attempt + 2}/3...")
                        time.sleep(1)
                        continue
                    else:
                        raise Exception(f"DeepSeek API connection error: {str(e)}")

        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API error: {str(e)}")

    def get_usage_stats(self) -> Dict[str, any]:
        """Get usage statistics"""
        return self.usage_stats.copy()

    def get_model_info(self) -> Dict[str, any]:
        """Get information about available models and clients"""
        return {
            "initialized_clients": list(self.clients.keys()),
            "available_models": [
                "gpt-4-turbo-preview",
                "claude-3-sonnet-20240229",
                "gemini-pro",
                "deepseek-chat"
            ]
        }