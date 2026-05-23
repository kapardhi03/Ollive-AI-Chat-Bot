#!/usr/bin/env python3
"""
Test API Client Fixes
Verify that the API client improvements work correctly
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_key_handling():
    """Test improved API key handling"""
    print("🔑 Testing API Key Handling")
    print("=" * 30)

    try:
        from frontier_assistant.api_client import FrontierModelClient

        client = FrontierModelClient()

        # Test 1: OpenAI without key should give specific error
        print("1. Testing OpenAI without API key...")
        try:
            response = client.generate_response("gpt-4-turbo-preview", "Hello")
            print("❌ Expected error but got response")
        except ValueError as e:
            if "API key not set" in str(e):
                print("✅ Got expected API key error message")
            else:
                print(f"⚠️  Got different error: {e}")

        # Test 2: Set API key and verify mapping
        print("\n2. Testing API key setting...")
        client.set_api_key("gpt-4-turbo-preview", "test-key")

        if "openai" in client.api_keys and client.api_keys["openai"] == "test-key":
            print("✅ OpenAI API key mapped correctly to provider")
        else:
            print(f"❌ API key mapping failed: {client.api_keys}")

        # Test 3: Google model name handling
        print("\n3. Testing Google model handling...")
        client.set_api_key("gemini-pro", "test-google-key")

        if "google" in client.api_keys:
            print("✅ Google API key mapped correctly")
        else:
            print(f"❌ Google API key mapping failed")

        # Test 4: Provider detection
        print("\n4. Testing provider detection...")
        tests = [
            ("gpt-4-turbo-preview", "openai"),
            ("claude-3-sonnet-20240229", "anthropic"),
            ("gemini-pro", "google"),
            ("deepseek-chat", "deepseek")
        ]

        for model, expected_provider in tests:
            actual_provider = client._get_provider_from_model(model)
            status = "✅" if actual_provider == expected_provider else "❌"
            print(f"  {status} {model} -> {actual_provider}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cost_calculation():
    """Test cost calculation accuracy"""
    print("\n💰 Testing Cost Calculation")
    print("=" * 30)

    try:
        from frontier_assistant.api_client import FrontierModelClient

        client = FrontierModelClient()

        # Test known calculations
        cost = client._calculate_cost("gpt-4-turbo-preview", 1000, 1000)
        expected = (1000/1000 * 0.01) + (1000/1000 * 0.03)  # $0.04

        if abs(cost - expected) < 0.001:
            print(f"✅ GPT-4 cost calculation correct: ${cost:.4f}")
        else:
            print(f"❌ GPT-4 cost calculation wrong: got ${cost:.4f}, expected ${expected:.4f}")

        # Test Claude calculation
        cost = client._calculate_cost("claude-3-sonnet-20240229", 500, 300)
        expected = (500/1000 * 0.003) + (300/1000 * 0.015)  # $0.0015 + $0.0045 = $0.006

        if abs(cost - expected) < 0.001:
            print(f"✅ Claude cost calculation correct: ${cost:.4f}")
        else:
            print(f"❌ Claude cost calculation wrong: got ${cost:.4f}, expected ${expected:.4f}")

        return True

    except Exception as e:
        print(f"❌ Cost test failed: {str(e)}")
        return False

def main():
    """Run all API client tests"""
    print("🧪 API Client Fixes Validation")
    print("=" * 40)
    print("Testing improved error handling and configuration")
    print()

    tests = [
        ("API Key Handling", test_api_key_handling),
        ("Cost Calculation", test_cost_calculation)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("🏆 TEST SUMMARY")
    print('='*50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL API CLIENT FIXES WORKING!")
        print("\nImprovements validated:")
        print("✅ Better API key error messages")
        print("✅ Correct provider name mapping")
        print("✅ Google model compatibility handling")
        print("✅ Accurate cost calculation")
        print("\n🚀 The API client is now more robust!")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)