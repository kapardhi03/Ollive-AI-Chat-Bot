#!/usr/bin/env python3
"""
Simple Google Model Tester
Test current Google model names to see which ones work
"""

import os

def test_google_models():
    """Test current Google model names"""
    print("🧪 Testing Current Google Model Names")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY=your_key_here")
        return False

    print(f"🔑 API key found (ending: ...{api_key[-4:]})")

    try:
        import google.generativeai as genai
        print("📦 Google generativeai package imported")
    except ImportError:
        print("❌ google-generativeai not installed")
        print("Install: pip install google-generativeai")
        return False

    # Configure API
    genai.configure(api_key=api_key)

    # Current 2025 model names to test
    models_to_test = [
        "gemini-2.0-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-pro"
    ]

    working_models = []
    test_prompt = "Say 'Hello' in one word."

    print(f"\n🔬 Testing {len(models_to_test)} models...")

    for model_name in models_to_test:
        print(f"\n📝 Testing: {model_name}")
        try:
            # Create model
            model = genai.GenerativeModel(model_name)

            # Generate content
            response = model.generate_content(
                test_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1
                )
            )

            if response.text and response.text.strip():
                result = response.text.strip()
                print(f"  ✅ SUCCESS: '{result}'")
                working_models.append(model_name)
            else:
                print(f"  ❌ Empty response")

        except Exception as e:
            error_str = str(e)
            if "404" in error_str:
                print(f"  ❌ Model not found")
            elif "permission" in error_str.lower():
                print(f"  ❌ Permission denied")
            elif "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"  ⚠️  Rate limit/quota issue")
            else:
                print(f"  ❌ Error: {error_str[:50]}...")

    # Results
    print(f"\n🎯 RESULTS")
    print("=" * 20)
    print(f"✅ Working models: {len(working_models)}")

    if working_models:
        print("\nWorking models:")
        for model in working_models:
            print(f"  ✅ {model}")

        print(f"\n📝 Update your scripts to use these model names:")
        print("Example:")
        print(f"  model = genai.GenerativeModel('{working_models[0]}')")

        return working_models
    else:
        print("\n❌ No models are working")
        print("\nPossible issues:")
        print("- API key lacks proper permissions")
        print("- Billing not enabled on Google Cloud project")
        print("- Regional restrictions")
        print("- API quota exceeded")
        return False

if __name__ == "__main__":
    result = test_google_models()
    if result:
        print(f"\n🎉 Success! Found {len(result)} working Google models")
    else:
        print(f"\n❌ No working models found")