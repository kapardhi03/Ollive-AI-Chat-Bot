#!/usr/bin/env python3
"""
Discover Available Google Models
Uses Google API to list current available models
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def discover_google_models():
    """Discover and test available Google models"""
    print("🔍 Discovering Google Gemini Models")
    print("=" * 40)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        print("Set with: export GOOGLE_API_KEY=your_key")
        return False

    try:
        # Try to import Google API
        try:
            import google.generativeai as genai
            print("📦 Using google.generativeai package")
        except ImportError:
            print("❌ google-generativeai package not installed")
            print("Install with: pip install google-generativeai")
            return False

        # Configure API
        genai.configure(api_key=api_key)
        print(f"🔑 API configured (key ends: ...{api_key[-4:]})")

        # List available models
        print("\n📋 Listing all available models...")
        try:
            models = genai.list_models()

            print("Available models:")
            working_models = []

            for model in models:
                model_name = model.name
                # Remove the 'models/' prefix if present
                clean_name = model_name.replace('models/', '')

                print(f"  📝 {clean_name}")

                # Check if it supports generateContent
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        working_models.append(clean_name)
                        print(f"    ✅ Supports generateContent")
                    else:
                        print(f"    ❌ Does not support generateContent")
                else:
                    # Try to use it anyway
                    working_models.append(clean_name)
                    print(f"    ⚠️  Support unknown, will test")

            print(f"\n🎯 Found {len(working_models)} potentially working models:")
            for model in working_models:
                print(f"  - {model}")

            return working_models

        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")

            # Try current 2025 model names as fallback
            print("\n🔄 Trying current 2025 model names as fallback...")
            common_models = [
                "gemini-2.0-flash",  # Current stable model
                "gemini-3.1-pro-preview",  # Latest flagship
                "gemini-3.1-flash-lite-preview",  # High-efficiency
                "gemini-2.5-flash",  # Stable alternative
                "gemini-1.5-flash",  # Legacy (might work)
                "gemini-pro",  # Legacy fallback
            ]

            return common_models

    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        return False

def test_models(model_names):
    """Test each model to see which ones actually work"""
    print(f"\n🧪 Testing {len(model_names)} models...")
    print("=" * 40)

    try:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        working_models = []
        test_prompt = "Hello, how are you?"

        for model_name in model_names:
            print(f"\n🔬 Testing: {model_name}")
            try:
                # Create model instance
                model = genai.GenerativeModel(model_name)

                # Try to generate content
                response = model.generate_content(
                    test_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=50,
                        temperature=0.1
                    )
                )

                if response.text and len(response.text.strip()) > 0:
                    print(f"  ✅ Working! Response: {response.text.strip()[:50]}...")
                    working_models.append(model_name)
                else:
                    print(f"  ❌ Empty response")

            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    print(f"  ❌ Model not found")
                elif "quota" in error_msg.lower():
                    print(f"  ⚠️  Quota/rate limit hit")
                elif "permission" in error_msg.lower():
                    print(f"  ⚠️  Permission denied")
                else:
                    print(f"  ❌ Error: {error_msg[:100]}")

        print(f"\n🎉 Successfully tested {len(working_models)} working models:")
        for model in working_models:
            print(f"  ✅ {model}")

        return working_models

    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        return []

def update_evaluation_script(working_models):
    """Update the evaluation script with working model names"""
    if not working_models:
        print("\n❌ No working models found, cannot update script")
        return False

    print(f"\n📝 Updating run_google_evaluation.py with working models...")

    try:
        # Read current script
        script_path = "run_google_evaluation.py"
        with open(script_path, 'r') as f:
            content = f.read()

        # Find and replace the models_to_test line
        old_line = 'models_to_test = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]'
        new_models = ', '.join([f'"{model}"' for model in working_models])
        new_line = f'models_to_test = [{new_models}]'

        if old_line in content:
            content = content.replace(old_line, new_line)

            # Write back
            with open(script_path, 'w') as f:
                f.write(content)

            print(f"✅ Updated script with {len(working_models)} working models")
            return True
        else:
            print("⚠️  Could not find models_to_test line in script")
            print(f"Manually update with: {new_line}")
            return False

    except Exception as e:
        print(f"❌ Error updating script: {str(e)}")
        return False

def main():
    """Main discovery and testing function"""
    print("🚀 Google Gemini Model Discovery & Testing")
    print("=" * 50)
    print("Finding models that work with your API key\n")

    # Step 1: Discover models
    model_names = discover_google_models()
    if not model_names:
        return False

    # Step 2: Test models
    working_models = test_models(model_names)
    if not working_models:
        print("\n❌ No models are working with your API key")
        print("\nPossible issues:")
        print("- API key doesn't have proper permissions")
        print("- Account needs billing enabled")
        print("- Geographic restrictions")
        print("- Quota exceeded")
        return False

    # Step 3: Update evaluation script
    update_evaluation_script(working_models)

    print(f"\n🎯 SUCCESS! Found {len(working_models)} working Google models")
    print("\n📋 Next steps:")
    print("1. Run updated evaluation: python run_google_evaluation.py")
    print("2. Or use any of these model names in your scripts:")
    for model in working_models:
        print(f"   - {model}")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)