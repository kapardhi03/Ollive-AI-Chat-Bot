"""
Tests for Frontier Model Client functionality
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontier_assistant.api_client import FrontierModelClient


class TestFrontierModelClient:
    """Test frontier model client functionality"""

    def test_client_initialization(self):
        """Test client initialization"""
        client = FrontierModelClient()

        assert client.api_keys == {}
        assert client.clients == {}
        assert "requests" in client.usage_stats
        assert "total_cost" in client.usage_stats

    def test_provider_detection(self):
        """Test provider detection from model names"""
        client = FrontierModelClient()

        assert client._get_provider_from_model("gpt-4-turbo-preview") == "openai"
        assert client._get_provider_from_model("claude-3-sonnet-20240229") == "anthropic"
        assert client._get_provider_from_model("gemini-pro") == "google"
        assert client._get_provider_from_model("deepseek-chat") == "deepseek"

        with pytest.raises(ValueError):
            client._get_provider_from_model("unknown-model")

    def test_cost_calculation(self):
        """Test cost calculation"""
        client = FrontierModelClient()

        # Test GPT-4 pricing
        cost = client._calculate_cost("gpt-4-turbo-preview", 1000, 500)
        expected = (1000 / 1000) * 0.01 + (500 / 1000) * 0.03  # $0.025
        assert abs(cost - expected) < 0.001

        # Test Claude pricing
        cost = client._calculate_cost("claude-3-sonnet-20240229", 1000, 500)
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015  # $0.0105
        assert abs(cost - expected) < 0.001

    def test_rate_limiting_reset(self):
        """Test rate limiting counter reset"""
        client = FrontierModelClient()

        # Simulate old request
        import time
        client.rate_limits["openai"]["last_request"] = time.time() - 70  # 70 seconds ago
        client.rate_limits["openai"]["request_count"] = 50

        # This should reset the counter
        client._check_rate_limit("openai")

        assert client.rate_limits["openai"]["request_count"] == 1

    def test_usage_stats_update(self):
        """Test usage statistics tracking"""
        client = FrontierModelClient()

        initial_requests = client.usage_stats["requests"]
        initial_cost = client.usage_stats["total_cost"]

        # Simulate usage update
        client._update_usage_stats("gpt-4-turbo-preview", "Hello world", "Hi there!", 1.5)

        assert client.usage_stats["requests"] == initial_requests + 1
        assert client.usage_stats["total_cost"] > initial_cost
        assert client.usage_stats["last_request"] is not None

    def test_get_usage_stats(self):
        """Test usage stats retrieval"""
        client = FrontierModelClient()

        stats = client.get_usage_stats()
        assert isinstance(stats, dict)
        assert "requests" in stats
        assert "tokens_used" in stats
        assert "total_cost" in stats

    def test_model_info(self):
        """Test model info retrieval"""
        client = FrontierModelClient()

        info = client.get_model_info()
        assert isinstance(info, dict)
        assert "initialized_clients" in info
        assert "available_models" in info
        assert isinstance(info["available_models"], list)

    @patch('frontier_assistant.api_client.openai.OpenAI')
    def test_openai_client_initialization(self, mock_openai):
        """Test OpenAI client initialization"""
        client = FrontierModelClient()
        client.set_api_key("gpt-4-turbo-preview", "test-key")

        mock_openai.assert_called_once_with(api_key="test-key")
        assert "openai" in client.clients

    @patch('frontier_assistant.api_client.Anthropic')
    def test_anthropic_client_initialization(self, mock_anthropic):
        """Test Anthropic client initialization"""
        client = FrontierModelClient()
        client.set_api_key("claude-3-sonnet-20240229", "test-key")

        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert "anthropic" in client.clients


class TestErrorHandling:
    """Test error handling in frontier client"""

    def test_invalid_model_error(self):
        """Test error handling for invalid model"""
        client = FrontierModelClient()

        with pytest.raises(ValueError):
            client._get_provider_from_model("invalid-model-name")

    def test_missing_client_error(self):
        """Test error handling for missing client"""
        client = FrontierModelClient()

        # Try to generate response without setting up client
        response = client.generate_response(
            "gpt-4-turbo-preview",
            "Hello",
            [],
            0.7,
            100,
            "default"
        )

        # Should return error message
        assert "error" in response.lower() or "apologize" in response.lower()

    def test_api_key_validation(self):
        """Test API key validation"""
        client = FrontierModelClient()

        # Test with empty API key
        try:
            client.set_api_key("gpt-4-turbo-preview", "")
            # Should either raise an error or handle gracefully
        except Exception:
            pass  # Expected behavior

    def test_rate_limit_handling(self):
        """Test rate limit handling"""
        client = FrontierModelClient()

        # Set artificial rate limit
        provider = "openai"
        client.rate_limits[provider]["request_count"] = 100  # Over limit
        client.rate_limits[provider]["requests_per_minute"] = 60

        # This should handle rate limiting gracefully
        try:
            client._check_rate_limit(provider)
        except Exception as e:
            pytest.fail(f"Rate limiting should be handled gracefully: {e}")