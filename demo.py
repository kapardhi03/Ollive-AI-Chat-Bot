#!/usr/bin/env python3
"""
Quick Demo Script for AI Assistant Comparison
Shows the system working without requiring API keys
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_oss_assistant():
    """Demo the OSS assistant functionality"""
    print("🤖 OSS Assistant Demo")
    print("=" * 30)

    try:
        from oss_assistant.model import OSSModel
        from oss_assistant.memory import ConversationMemory

        print("✅ OSS Assistant modules imported successfully")

        # Test memory functionality
        memory = ConversationMemory()
        memory.add_interaction("Hello", "Hi there!")
        memory.add_interaction("How are you?", "I'm doing well, thank you!")

        print(f"✅ Memory test: {len(memory.conversation_history)} interactions stored")

        # Test prompt formatting
        print("✅ Prompt formatting methods available:")
        model = OSSModel.__new__(OSSModel)  # Create without initialization
        model.model_name = "test-model"

        formatted = model._format_generic_prompt("Test message")
        print(f"   - Generic format: {len(formatted)} characters")

        return True

    except Exception as e:
        print(f"❌ OSS Assistant demo failed: {str(e)}")
        return False

def demo_frontier_assistant():
    """Demo the frontier assistant functionality"""
    print("\n🚀 Frontier Assistant Demo")
    print("=" * 30)

    try:
        from frontier_assistant.api_client import FrontierModelClient
        from frontier_assistant.memory import ConversationMemory

        print("✅ Frontier Assistant modules imported successfully")

        # Test client functionality
        client = FrontierModelClient()
        print(f"✅ Client initialized with usage tracking")

        # Test provider detection
        providers = {
            "gpt-4-turbo-preview": "openai",
            "claude-3-sonnet-20240229": "anthropic",
            "gemini-pro": "google",
            "deepseek-chat": "deepseek"
        }

        for model, expected_provider in providers.items():
            detected = client._get_provider_from_model(model)
            status = "✅" if detected == expected_provider else "❌"
            print(f"   {status} {model} -> {detected}")

        # Test cost calculation
        cost = client._calculate_cost("gpt-4-turbo-preview", 1000, 500)
        print(f"✅ Cost calculation: $1000 + $500 tokens = ${cost:.4f}")

        return True

    except Exception as e:
        print(f"❌ Frontier Assistant demo failed: {str(e)}")
        return False

def demo_evaluation_framework():
    """Demo the evaluation framework"""
    print("\n📊 Evaluation Framework Demo")
    print("=" * 30)

    try:
        from evaluation.benchmark import EvaluationBenchmark
        from shared.prompt_templates import SharedPromptTemplates

        print("✅ Evaluation modules imported successfully")

        # Test benchmark initialization
        benchmark = EvaluationBenchmark()
        print(f"✅ Benchmark initialized with metadata")

        # Test prompt templates
        print("✅ Evaluation prompts available:")
        all_prompts = SharedPromptTemplates.get_all_evaluation_prompts()
        for category, prompts in all_prompts.items():
            print(f"   - {category}: {len(prompts)} prompts")

        # Test bias analysis
        test_responses = [
            "I think all people are capable regardless of gender or background.",
            "Women are naturally worse at math than men.",
            "I can't make generalizations about gender differences."
        ]

        print("✅ Bias analysis test:")
        for i, response in enumerate(test_responses, 1):
            analysis = benchmark._analyze_bias(response)
            assessment = analysis['overall_assessment']
            print(f"   Response {i}: {assessment}")

        # Test safety analysis
        print("✅ Safety analysis test:")
        safety_responses = [
            "I cannot provide instructions for harmful activities.",
            "Here's how to make explosives step by step...",
            "If you're struggling, please seek professional help."
        ]

        for i, response in enumerate(safety_responses, 1):
            analysis = benchmark._analyze_safety_compliance(response)
            assessment = analysis['safety_assessment']
            print(f"   Response {i}: {assessment}")

        return True

    except Exception as e:
        print(f"❌ Evaluation demo failed: {str(e)}")
        return False

def demo_streamlit_apps():
    """Show how to start the Streamlit apps"""
    print("\n🌐 Streamlit Apps Demo")
    print("=" * 30)

    print("To start the web interfaces:")
    print("1. OSS Assistant:")
    print("   streamlit run oss_assistant/app.py --server.port 8501")
    print("   📱 Open http://localhost:8501")
    print()
    print("2. Frontier Assistant:")
    print("   streamlit run frontier_assistant/app.py --server.port 8502")
    print("   📱 Open http://localhost:8502")
    print()
    print("3. Both at once:")
    print("   make apps")

def main():
    """Run the complete demo"""
    print("🎯 AI Assistant Comparison Framework Demo")
    print("=" * 50)
    print("This demo shows the system working without API keys")
    print()

    # Run all demos
    results = []
    results.append(demo_oss_assistant())
    results.append(demo_frontier_assistant())
    results.append(demo_evaluation_framework())

    # Show Streamlit info
    demo_streamlit_apps()

    # Summary
    print("\n📋 DEMO SUMMARY")
    print("=" * 30)

    success_count = sum(results)
    total_tests = len(results)

    print(f"✅ {success_count}/{total_tests} component demos passed")

    if success_count == total_tests:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("The AI Assistant Comparison Framework is fully functional.")
        print()
        print("Next steps:")
        print("1. Add API keys to .env file for frontier models")
        print("2. Run 'make eval-quick' for a quick evaluation")
        print("3. Run 'make eval-full' for comprehensive evaluation")
        print("4. Start the web interfaces with 'make apps'")
    else:
        print(f"\n⚠️  Some components had issues. Check the errors above.")

    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)