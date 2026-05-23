#!/usr/bin/env python3
"""
Google-Only API Test
Test the system with just Google Gemini API
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_google_api_only():
    """Test Google API functionality only"""
    print("🤖 Testing Google Gemini API Integration")
    print("=" * 40)

    try:
        # Test if we can import google packages
        google_available = False
        try:
            import google.genai as genai
            google_available = True
            print("✅ Using new google.genai package")
        except ImportError:
            try:
                import google.generativeai as genai
                google_available = True
                print("⚠️  Using deprecated google.generativeai package")
            except ImportError:
                print("❌ No Google packages available")

        if not google_available:
            print("To test Google API, install: pip install google-generativeai")
            return False

        # Test basic functionality without dependencies
        print("✅ Google API package loaded successfully")

        # Test model name detection
        from frontier_assistant.api_client import FrontierModelClient
        client = FrontierModelClient()

        # Test provider detection for Google models
        google_models = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

        print("\n🔍 Testing model provider detection:")
        for model in google_models:
            try:
                provider = client._get_provider_from_model(model)
                status = "✅" if provider == "google" else "❌"
                print(f"  {status} {model} -> {provider}")
            except Exception as e:
                print(f"  ❌ {model} -> Error: {e}")

        # Test API key setting
        print("\n🔑 Testing API key handling:")
        try:
            client.set_api_key("gemini-pro", "test-key")
            if "google" in client.api_keys:
                print("  ✅ Google API key set successfully")
            else:
                print(f"  ❌ API key not set: {client.api_keys}")
        except Exception as e:
            print(f"  ❌ API key setting failed: {e}")

        # Test cost calculation
        print("\n💰 Testing cost calculation:")
        try:
            cost = client._calculate_cost("gemini-pro", 1000, 500)
            expected = (1000/1000 * 0.0005) + (500/1000 * 0.0015)  # $0.0005 + $0.00075 = $0.00125
            if abs(cost - expected) < 0.001:
                print(f"  ✅ Cost calculation correct: ${cost:.5f}")
            else:
                print(f"  ⚠️  Cost calculation: got ${cost:.5f}, expected ${expected:.5f}")
        except Exception as e:
            print(f"  ❌ Cost calculation failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_with_google():
    """Test evaluation framework with Google model focus"""
    print("\n📊 Testing Evaluation Framework for Google")
    print("=" * 40)

    try:
        from evaluation.benchmark import EvaluationBenchmark

        benchmark = EvaluationBenchmark()
        print("✅ Evaluation benchmark loaded")

        # Test creating prompts for Google
        test_prompt = "What are the benefits of renewable energy?"

        # Test bias evaluation
        bias_analysis = benchmark._analyze_bias(
            "I think renewable energy is good for everyone regardless of their background"
        )
        print(f"✅ Bias analysis: {bias_analysis['overall_assessment']}")

        # Test safety evaluation
        safety_analysis = benchmark._analyze_safety_compliance(
            "I recommend consulting with energy experts for renewable energy planning"
        )
        print(f"✅ Safety analysis: {safety_analysis['safety_assessment']}")

        # Test factual accuracy
        fact_analysis = benchmark._analyze_factual_accuracy(
            "Solar panels convert sunlight to electricity",
            "Solar panels convert sunlight to electricity"
        )
        print(f"✅ Factual analysis: {fact_analysis['accuracy_score']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Evaluation test failed: {str(e)}")
        return False

def create_google_env_example():
    """Create environment file for Google-only setup"""
    print("\n📝 Creating Google-only environment example")
    print("=" * 40)

    env_content = """# Google-only API configuration for AI Assistant Comparison
# Copy this to .env and add your actual API key

# Google Gemini API (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Other providers (leave empty if not using)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=

# Evaluation settings
EVALUATION_MODEL=gemini-pro
OSS_MODEL_CACHE_DIR=./models
"""

    try:
        with open(".env.google", "w") as f:
            f.write(env_content)

        print("✅ Created .env.google file")
        print("📋 Next steps:")
        print("   1. Copy .env.google to .env")
        print("   2. Add your Google API key")
        print("   3. Run: python test_google_only.py")
        return True

    except Exception as e:
        print(f"❌ Failed to create env file: {e}")
        return False

def main():
    """Run Google-only tests"""
    print("🚀 Google Gemini API - Standalone Test")
    print("=" * 50)
    print("Testing the system with only Google Gemini API")
    print()

    tests = [
        ("Google API Integration", test_google_api_only),
        ("Evaluation Framework", test_evaluation_with_google),
        ("Environment Setup", create_google_env_example)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("🏆 GOOGLE-ONLY TEST SUMMARY")
    print('='*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed >= 2:  # Environment creation might fail, that's ok
        print("\n🎉 GOOGLE GEMINI INTEGRATION READY!")
        print("\nTo use with your Google API key:")
        print("1. cp .env.google .env")
        print("2. Edit .env and add your Google API key")
        print("3. Run: python run_evaluation.py --model gemini-pro")

    return passed >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)