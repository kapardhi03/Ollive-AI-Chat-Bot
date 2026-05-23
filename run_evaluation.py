#!/usr/bin/env python3
"""
Main evaluation runner script
Execute complete evaluation of both OSS and Frontier assistants
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.benchmark import EvaluationBenchmark
from oss_assistant.model import OSSModel
from frontier_assistant.api_client import FrontierModelClient

def check_requirements():
    """Check if all requirements are met"""
    requirements_ok = True

    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }

    print("🔍 Checking environment setup...")

    missing_keys = [key for key, value in api_keys.items() if not value]
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("   Set them as environment variables or provide them interactively")
        requirements_ok = False
    else:
        print("✅ API keys found in environment")

    # Check Python packages
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found - required for OSS models")
        requirements_ok = False

    try:
        import streamlit
        print(f"✅ Streamlit available: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found - required for web interface")
        requirements_ok = False

    return requirements_ok

def setup_api_keys(client):
    """Interactively setup API keys"""
    api_configs = [
        ("OpenAI GPT models", "gpt-4-turbo-preview", "OPENAI_API_KEY"),
        ("Anthropic Claude models", "claude-3-sonnet-20240229", "ANTHROPIC_API_KEY"),
        ("Google Gemini models", "gemini-pro", "GOOGLE_API_KEY")
    ]

    print("\n🔑 API Key Setup")
    print("You can skip providers you don't want to test (press Enter)")

    for provider_name, model_id, env_var in api_configs:
        existing_key = os.getenv(env_var)

        if existing_key:
            print(f"✅ {provider_name}: Found in environment")
            client.set_api_key(model_id, existing_key)
        else:
            key = input(f"Enter API key for {provider_name} (or press Enter to skip): ").strip()
            if key:
                try:
                    client.set_api_key(model_id, key)
                    print(f"✅ {provider_name}: Configured")
                except Exception as e:
                    print(f"❌ {provider_name}: Configuration failed - {str(e)}")

def run_evaluation(oss_model_name="microsoft/Phi-3-mini-4k-instruct",
                   frontier_model="gpt-4-turbo-preview",
                   quick_mode=False):
    """Run the complete evaluation"""

    print(f"\n🚀 Starting AI Assistant Evaluation")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 OSS Model: {oss_model_name}")
    print(f"🌐 Frontier Model: {frontier_model}")

    if quick_mode:
        print("⚡ Quick mode: Limited prompt set")

    try:
        # Initialize components
        print("\n📦 Loading OSS model...")
        oss_model = OSSModel(oss_model_name)

        print("🔌 Setting up frontier client...")
        frontier_client = FrontierModelClient()
        setup_api_keys(frontier_client)

        print("📊 Initializing evaluation benchmark...")
        benchmark = EvaluationBenchmark()

        # Run evaluation
        print("\n🧪 Running evaluation suite...")
        if quick_mode:
            # Run limited evaluation for testing
            results = benchmark.run_quick_evaluation(oss_model, frontier_client, frontier_model)
        else:
            results = benchmark.run_full_evaluation(oss_model, frontier_client, frontier_model)

        # Save and display results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation/results/evaluation_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {results_file}")

        # Display summary
        if "summary" in results:
            print("\n📋 SUMMARY RESULTS")
            print("=" * 50)
            summary = results["summary"]

            print("Overall Scores:")
            for assistant_type in ["oss", "frontier"]:
                if assistant_type in summary["overall_scores"]:
                    scores = summary["overall_scores"][assistant_type]
                    print(f"  {assistant_type.upper()} Assistant: {scores.get('overall', 0):.2f}")

            print("\nRecommendations:")
            for rec in summary.get("recommendations", []):
                print(f"  • {rec}")

        return results

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run AI Assistant Evaluation")
    parser.add_argument("--oss-model", default="microsoft/Phi-3-mini-4k-instruct",
                       help="OSS model to evaluate")
    parser.add_argument("--frontier-model", default="gpt-4-turbo-preview",
                       help="Frontier model to evaluate")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick evaluation with limited prompts")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip environment checks")

    args = parser.parse_args()

    print("🔬 AI Assistant Evaluation Framework")
    print("=" * 40)

    if not args.skip_checks:
        if not check_requirements():
            print("\n❌ Environment check failed. Please install requirements first:")
            print("pip install -r requirements.txt")
            return 1

    # Create necessary directories
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Run evaluation
    results = run_evaluation(
        oss_model_name=args.oss_model,
        frontier_model=args.frontier_model,
        quick_mode=args.quick
    )

    if results:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"🔍 View detailed results in Jupyter notebook: evaluation/report.ipynb")
        print(f"🌐 Start web interfaces:")
        print(f"   OSS Assistant: streamlit run oss_assistant/app.py --server.port 8501")
        print(f"   Frontier Assistant: streamlit run frontier_assistant/app.py --server.port 8502")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())