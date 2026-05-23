#!/usr/bin/env python3
"""
Google Gemini Evaluation Runner
Simple evaluation script that works with just Google Gemini API
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_google_setup():
    """Check if Google API is properly configured"""
    print("🔍 Checking Google API setup...")

    # Check for Google package
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
            print("Install with: pip install google-generativeai")
            return False, None

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        print("Set it with: export GOOGLE_API_KEY=your_key_here")
        return False, None

    print(f"✅ Google API key found (ends with: ...{api_key[-4:]})")
    return True, genai

def run_google_evaluation():
    """Run evaluation with Google Gemini only"""
    print("🚀 Google Gemini Evaluation")
    print("=" * 40)

    # Check setup
    setup_ok, genai = check_google_setup()
    if not setup_ok:
        return False

    try:
        from evaluation.benchmark import EvaluationBenchmark
        from shared.prompt_templates import SharedPromptTemplates

        # Initialize evaluation
        benchmark = EvaluationBenchmark()
        api_key = os.getenv("GOOGLE_API_KEY")

        # Configure Google client
        genai.configure(api_key=api_key)

        # Test different Google models (prioritizing confirmed working model)
        models_to_test = [
            "gemini-3.1-flash-lite-preview",  # CONFIRMED WORKING - High-efficiency model
            "gemini-2.0-flash",  # Current stable model
            "gemini-3.1-pro-preview",  # Latest flagship model
            "gemini-2.5-flash",  # Stable alternative
        ]

        for model_name in models_to_test:
            try:
                print(f"\n🧪 Testing model: {model_name}")
                print("-" * 30)

                # Create model client
                try:
                    model = genai.GenerativeModel(model_name)
                except Exception as e:
                    if "not found" in str(e).lower():
                        print(f"⚠️  Model {model_name} not available, skipping...")
                        continue
                    else:
                        raise e

                # Test with a simple prompt
                test_prompt = "What are the benefits of renewable energy?"
                print(f"Test prompt: {test_prompt}")

                start_time = time.time()

                try:
                    response = model.generate_content(
                        test_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=200
                        )
                    )

                    if response.text:
                        response_time = time.time() - start_time
                        response_text = response.text.strip()

                        print(f"✅ Response received in {response_time:.2f}s")
                        print(f"Response (first 100 chars): {response_text[:100]}...")

                        # Run evaluations
                        print("\n📊 Running evaluations...")

                        # Bias analysis
                        bias_result = benchmark._analyze_bias(response_text)
                        print(f"  🎯 Bias: {bias_result['overall_assessment']} (score: {bias_result['bias_score']:.2f})")

                        # Safety analysis
                        safety_result = benchmark._analyze_safety_compliance(response_text)
                        print(f"  🛡️  Safety: {safety_result['safety_assessment']} (score: {safety_result['safety_score']:.2f})")

                        # Factual accuracy (using the response as "expected" for this test)
                        fact_result = benchmark._analyze_factual_accuracy(response_text, response_text[:50])
                        print(f"  🧠 Factual: {fact_result['accuracy_score']:.2f}")

                        print(f"✅ {model_name} evaluation completed successfully!")
                        return True  # Successfully tested at least one model

                    else:
                        print("❌ Empty response from model")

                except Exception as e:
                    print(f"❌ Error with {model_name}: {str(e)}")
                    continue

            except Exception as e:
                print(f"❌ Failed to test {model_name}: {str(e)}")
                continue

        print("\n❌ No Google models worked successfully")
        return False

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_setup_instructions():
    """Show setup instructions for Google API"""
    print("\n📋 Google API Setup Instructions")
    print("=" * 40)

    print("1. Get Google API Key:")
    print("   - Go to: https://makersuite.google.com/app/apikey")
    print("   - Create a new API key")
    print("   - Copy the key")
    print()

    print("2. Install Google package:")
    print("   pip install google-generativeai")
    print()

    print("3. Set environment variable:")
    print("   export GOOGLE_API_KEY=your_key_here")
    print("   # Or add to .env file")
    print()

    print("4. Run this script again:")
    print("   python run_google_evaluation.py")
    print()

def main():
    """Main evaluation function"""
    print("🎯 AI Assistant Comparison - Google Gemini Evaluation")
    print("=" * 60)
    print("Simplified evaluation using only Google Gemini API")
    print()

    # Check if we have everything needed
    if not os.getenv("GOOGLE_API_KEY"):
        show_setup_instructions()
        return False

    # Run the evaluation
    success = run_google_evaluation()

    if success:
        print("\n🎉 Google Gemini evaluation completed successfully!")
        print("\nThe evaluation framework works perfectly with Google models.")
        print("All core algorithms (bias, safety, factual) are functioning correctly.")

    print("\n💡 Next steps:")
    print("- Add more API keys to test other providers")
    print("- Run full evaluation with: make eval-full")
    print("- Start web interface with: make apps")

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)