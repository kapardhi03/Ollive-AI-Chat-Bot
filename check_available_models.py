#!/usr/bin/env python3
"""
Check Available Google Gemini Models
List what models are accessible with your API key
"""

import os
import google.generativeai as genai

def check_available_models():
    """Check what models are available with the current API key"""

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return False

    print("🔍 Checking Available Google Gemini Models")
    print("=" * 50)

    # Configure API
    genai.configure(api_key=api_key)

    try:
        # List available models
        print("📋 Available Models:")
        models = genai.list_models()

        available_models = []
        for model in models:
            print(f"  • {model.name}")
            print(f"    Display Name: {model.display_name}")
            print(f"    Supported Methods: {', '.join(model.supported_generation_methods)}")
            print()

            # Check if it supports generateContent
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)

        print("✅ Models Supporting generateContent:")
        for model_name in available_models:
            print(f"  • {model_name}")

        print(f"\n📊 Total: {len(available_models)} models available for text generation")

        # Test the most common model
        if available_models:
            print(f"\n🧪 Testing first available model: {available_models[0]}")
            try:
                test_model = genai.GenerativeModel(available_models[0])
                response = test_model.generate_content("Hello, how are you?")
                print(f"✅ Test successful: {response.text[:50]}...")
                return available_models
            except Exception as e:
                print(f"❌ Test failed: {e}")
                return []
        else:
            print("❌ No models support generateContent")
            return []

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

if __name__ == "__main__":
    models = check_available_models()
    if models:
        print(f"\n🎯 Recommendation: Use '{models[0]}' in your evaluation script")
    else:
        print("\n💥 No compatible models found")