#!/usr/bin/env python3
"""
Google Gemini Models Evaluation Demo
Simplified evaluation comparing two Google Gemini models

Usage:
    export GOOGLE_API_KEY='your-google-api-key-here'
    python demo_google_evaluation.py
"""

import asyncio
import os
import json
from datetime import datetime
from pathlib import Path

# Ensure we can import our evaluation modules
import sys
sys.path.append(str(Path(__file__).parent))

from evaluation.comprehensive_assistant_evaluation import ComprehensiveAssistantEvaluator


def main():
    """Main function for Google-only evaluation"""

    print("🚀 Google Gemini Models Evaluation Demo")
    print("=" * 50)
    print()
    print("This demo will compare Google Gemini models on:")
    print("  🎯 Hallucination Rate (factual accuracy)")
    print("  ⚖️ Bias & Harmful Outputs (fairness)")
    print("  🛡️ Content Safety (jailbreak resistance)")
    print()

    # Check for Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("❌ Missing Google API Key")
        print()
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print()
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return 1

    print("✅ Google API key found")
    print()

    # Model configurations
    print("🤖 Comparing Google Gemini Models:")
    print("  📊 Gemini 1.5 Pro vs Gemini 1.5 Flash")
    print("  📝 Testing on factual accuracy, bias resistance, and safety")
    print()

    # Run the evaluation
    async def run_evaluation():
        evaluator = ComprehensiveAssistantEvaluator()

        assistant1_config = {
            "provider": "google",
            "model": "gemini-1.5-pro"
        }

        assistant2_config = {
            "provider": "google",
            "model": "gemini-1.5-flash"
        }

        print("⏱️ Starting evaluation (this may take a few minutes)...")
        print("📋 Running quick evaluation with key test cases...")
        print()

        results = await evaluator.evaluate_assistants(
            assistant1_config=assistant1_config,
            assistant2_config=assistant2_config,
            evaluation_mode="quick",  # Use quick mode for demo
            use_llm_judge=False,      # Disable LLM judge to avoid needing multiple APIs
            output_dir="google_demo_results"
        )

        return results

    try:
        # Run the evaluation
        results = asyncio.run(run_evaluation())

        # Display simplified results
        print_google_results(results)

        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Detailed results saved to: google_demo_results/")
        print()
        print("🔍 Next steps:")
        print("  📊 Run full evaluation: python evaluation/comprehensive_assistant_evaluation.py \\")
        print("      --assistant1 google:gemini-1.5-pro --assistant2 google:gemini-1.5-flash")
        print("  🌐 Launch dashboard: streamlit run evaluation/analysis_dashboard.py")

        return 0

    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        print("\nPossible issues:")
        print("  - Invalid Google API key")
        print("  - Network connectivity problems")
        print("  - API rate limits")
        print("  - Model availability")
        return 1


def print_google_results(results):
    """Print results in a user-friendly format"""

    print("\n📊 GOOGLE GEMINI EVALUATION RESULTS")
    print("=" * 60)

    try:
        comparative = results["evaluation_results"]["comparative_scores"]

        # Model names
        model1 = "Gemini 1.5 Pro"
        model2 = "Gemini 1.5 Flash"

        # Extract scores
        scores1 = comparative["assistant1_scores"]
        scores2 = comparative["assistant2_scores"]

        print(f"\n🏆 OVERALL WINNER: {comparative['overall_winner'].replace('assistant1', model1).replace('assistant2', model2)}")
        print(f"📈 Performance Gap: {comparative['performance_gap']*100:.1f} percentage points")

        print(f"\n📋 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<25} {model1:<20} {model2:<20} {'Winner':<15}")
        print("-" * 85)

        # Factual Accuracy
        acc1 = scores1["factual_accuracy"] * 100
        acc2 = scores2["factual_accuracy"] * 100
        winner_acc = model1 if acc1 > acc2 else model2 if acc2 > acc1 else "Tie"
        print(f"{'Factual Accuracy':<25} {acc1:<19.1f}% {acc2:<19.1f}% {winner_acc:<15}")

        # Bias Resistance
        bias1 = scores1["bias_resistance"] * 100
        bias2 = scores2["bias_resistance"] * 100
        winner_bias = model1 if bias1 > bias2 else model2 if bias2 > bias1 else "Tie"
        print(f"{'Bias Resistance':<25} {bias1:<19.1f}% {bias2:<19.1f}% {winner_bias:<15}")

        # Safety Compliance
        safety1 = scores1["safety_compliance"] * 100
        safety2 = scores2["safety_compliance"] * 100
        winner_safety = model1 if safety1 > safety2 else model2 if safety2 > safety1 else "Tie"
        print(f"{'Safety Compliance':<25} {safety1:<19.1f}% {safety2:<19.1f}% {winner_safety:<15}")

        # Composite Score
        comp1 = scores1["composite_score"] * 100
        comp2 = scores2["composite_score"] * 100
        winner_comp = model1 if comp1 > comp2 else model2 if comp2 > comp1 else "Tie"
        print(f"{'Composite Score':<25} {comp1:<19.1f}% {comp2:<19.1f}% {winner_comp:<15}")

        # Key insights
        print(f"\n🔍 KEY INSIGHTS:")

        # Hallucination analysis
        hallucination = results["evaluation_results"]["hallucination_assessment"]
        hall_rate1 = hallucination["assistant1_results"]["hallucination_rate"] * 100
        hall_rate2 = hallucination["assistant2_results"]["hallucination_rate"] * 100

        if abs(hall_rate1 - hall_rate2) > 5:
            better_model = model1 if hall_rate1 < hall_rate2 else model2
            difference = abs(hall_rate1 - hall_rate2)
            print(f"  🎯 {better_model} has {difference:.1f}% lower hallucination rate")
        else:
            print(f"  🎯 Both models show similar factual accuracy (~{(hall_rate1 + hall_rate2)/2:.1f}% error rate)")

        # Safety analysis
        safety = results["evaluation_results"]["safety_assessment"]
        violations1 = safety["assistant1_results"]["safety_violations"]
        violations2 = safety["assistant2_results"]["safety_violations"]
        total_prompts = safety["total_prompts"]

        print(f"  🛡️ {model1}: {violations1}/{total_prompts} safety violations")
        print(f"  🛡️ {model2}: {violations2}/{total_prompts} safety violations")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if results.get("recommendations"):
            for i, rec in enumerate(results["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✅ Both models perform well across all evaluation criteria")

    except Exception as e:
        print(f"Error displaying results: {e}")
        print(f"Raw results available in output files")


if __name__ == "__main__":
    exit(main())