#!/usr/bin/env python3
"""
Demo: AI Assistant Evaluation Comparison
Demonstrates comprehensive evaluation of assistants on hallucination, bias, and safety

This demo shows how to use the evaluation framework to compare two AI assistants
across the three critical dimensions: hallucination rate, bias & harmful outputs,
and content safety.

Usage:
    python demo_evaluation_comparison.py
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


def setup_demo_environment():
    """Setup demo environment with Google API key"""
    print("🔧 Setting up demo environment...")

    # Check for Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("⚠️ Missing API key: GOOGLE_API_KEY")
        print("Please set the Google API key to run the demo.")
        print("\nExample:")
        print("export GOOGLE_API_KEY='your-google-api-key-here'")
        print("\nYou can get a Google API key from: https://makersuite.google.com/app/apikey")
        return False

    print("✅ Google API key found")
    return True


async def run_demo_evaluation():
    """Run a demonstration evaluation comparing two assistants"""

    print("\n🔬 Starting AI Assistant Evaluation Demo")
    print("=" * 60)

    # Configuration for the demo - comparing two Google models
    assistant1_config = {
        "provider": "google",
        "model": "gemini-1.5-pro"
    }

    assistant2_config = {
        "provider": "google",
        "model": "gemini-1.5-flash"
    }

    print(f"🤖 Comparing:")
    print(f"  Assistant 1: {assistant1_config['provider'].title()} {assistant1_config['model']}")
    print(f"  Assistant 2: {assistant2_config['provider'].title()} {assistant2_config['model']}")
    print()

    # Initialize evaluator
    evaluator = ComprehensiveAssistantEvaluator()

    # Run evaluation in quick mode for demo
    results = await evaluator.evaluate_assistants(
        assistant1_config=assistant1_config,
        assistant2_config=assistant2_config,
        evaluation_mode="quick",  # Use quick mode for demo
        use_llm_judge=True,
        output_dir="demo_results"
    )

    return results


def analyze_demo_results(results):
    """Analyze and display demo results in a user-friendly way"""

    print("\n📊 DEMO EVALUATION RESULTS ANALYSIS")
    print("=" * 60)

    # Extract key metrics
    comparative = results["evaluation_results"]["comparative_scores"]
    hallucination = results["evaluation_results"]["hallucination_assessment"]
    bias = results["evaluation_results"]["bias_assessment"]
    safety = results["evaluation_results"]["safety_assessment"]

    # Display overall winner
    winner = comparative["overall_winner"]
    gap = comparative["performance_gap"]

    print(f"\n🏆 OVERALL WINNER: {winner.replace('assistant', 'Assistant')}")
    print(f"📈 Performance Gap: {gap*100:.1f} percentage points")

    # Detailed breakdown
    print(f"\n📋 DETAILED PERFORMANCE BREAKDOWN:")
    print(f"{'Metric':<25} {'Assistant 1':<15} {'Assistant 2':<15} {'Better':<10}")
    print("-" * 70)

    # Factual Accuracy (lower hallucination is better)
    acc1 = comparative["assistant1_scores"]["factual_accuracy"] * 100
    acc2 = comparative["assistant2_scores"]["factual_accuracy"] * 100
    better_acc = "Assistant 1" if acc1 > acc2 else "Assistant 2" if acc2 > acc1 else "Tie"
    print(f"{'Factual Accuracy':<25} {acc1:<14.1f}% {acc2:<14.1f}% {better_acc:<10}")

    # Bias Resistance (lower bias is better)
    bias1 = comparative["assistant1_scores"]["bias_resistance"] * 100
    bias2 = comparative["assistant2_scores"]["bias_resistance"] * 100
    better_bias = "Assistant 1" if bias1 > bias2 else "Assistant 2" if bias2 > bias1 else "Tie"
    print(f"{'Bias Resistance':<25} {bias1:<14.1f}% {bias2:<14.1f}% {better_bias:<10}")

    # Safety Compliance (higher safety is better)
    safety1 = comparative["assistant1_scores"]["safety_compliance"] * 100
    safety2 = comparative["assistant2_scores"]["safety_compliance"] * 100
    better_safety = "Assistant 1" if safety1 > safety2 else "Assistant 2" if safety2 > safety1 else "Tie"
    print(f"{'Safety Compliance':<25} {safety1:<14.1f}% {safety2:<14.1f}% {better_safety:<10}")

    # Composite Score
    comp1 = comparative["assistant1_scores"]["composite_score"] * 100
    comp2 = comparative["assistant2_scores"]["composite_score"] * 100
    better_comp = "Assistant 1" if comp1 > comp2 else "Assistant 2" if comp2 > comp1 else "Tie"
    print(f"{'Composite Score':<25} {comp1:<14.1f}% {comp2:<14.1f}% {better_comp:<10}")

    # Specific insights
    print(f"\n🔍 KEY INSIGHTS:")

    # Hallucination insights
    hall_rate1 = hallucination["assistant1_results"]["hallucination_rate"] * 100
    hall_rate2 = hallucination["assistant2_results"]["hallucination_rate"] * 100

    if hall_rate1 < hall_rate2:
        print(f"  🎯 Assistant 1 has {hall_rate2 - hall_rate1:.1f}% lower hallucination rate")
    elif hall_rate2 < hall_rate1:
        print(f"  🎯 Assistant 2 has {hall_rate1 - hall_rate2:.1f}% lower hallucination rate")
    else:
        print(f"  🎯 Both assistants have similar hallucination rates (~{hall_rate1:.1f}%)")

    # Safety insights
    safety_violations1 = safety["assistant1_results"]["safety_violations"]
    safety_violations2 = safety["assistant2_results"]["safety_violations"]
    total_safety_prompts = safety["total_prompts"]

    print(f"  🛡️ Assistant 1: {safety_violations1}/{total_safety_prompts} safety violations")
    print(f"  🛡️ Assistant 2: {safety_violations2}/{total_safety_prompts} safety violations")

    # Bias insights
    bias_responses1 = bias["assistant1_results"]["biased_responses"]
    bias_responses2 = bias["assistant2_results"]["biased_responses"]
    total_bias_prompts = bias["total_prompts"]

    print(f"  ⚖️ Assistant 1: {bias_responses1}/{total_bias_prompts} biased responses")
    print(f"  ⚖️ Assistant 2: {bias_responses2}/{total_bias_prompts} biased responses")

    # Top recommendations
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(results["recommendations"][:3], 1):
        print(f"  {i}. {rec}")

    print(f"\n📁 Full results saved to: demo_results/")


def create_demo_summary_report(results):
    """Create a markdown summary report of the demo results"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# AI Assistant Evaluation Demo Report

Generated: {timestamp}

## Evaluation Overview

This demonstration compared two AI assistants across three critical dimensions:

1. **Hallucination Rate** - Tendency to generate factually incorrect information
2. **Bias & Harmful Outputs** - Production of biased or discriminatory content
3. **Content Safety** - Resistance to jailbreaking and harmful content generation

## Assistant Configurations

- **Assistant 1**: {results['assistant_configs']['assistant1']['provider'].title()} {results['assistant_configs']['assistant1']['model']}
- **Assistant 2**: {results['assistant_configs']['assistant2']['provider'].title()} {results['assistant_configs']['assistant2']['model']}

## Results Summary

"""

    comparative = results["evaluation_results"]["comparative_scores"]

    # Add results table
    report += f"""
### Performance Scores

| Metric | Assistant 1 | Assistant 2 | Winner |
|--------|-------------|-------------|---------|
| Factual Accuracy | {comparative['assistant1_scores']['factual_accuracy']*100:.1f}% | {comparative['assistant2_scores']['factual_accuracy']*100:.1f}% | {'Assistant 1' if comparative['assistant1_scores']['factual_accuracy'] > comparative['assistant2_scores']['factual_accuracy'] else 'Assistant 2' if comparative['assistant2_scores']['factual_accuracy'] > comparative['assistant1_scores']['factual_accuracy'] else 'Tie'} |
| Bias Resistance | {comparative['assistant1_scores']['bias_resistance']*100:.1f}% | {comparative['assistant2_scores']['bias_resistance']*100:.1f}% | {'Assistant 1' if comparative['assistant1_scores']['bias_resistance'] > comparative['assistant2_scores']['bias_resistance'] else 'Assistant 2' if comparative['assistant2_scores']['bias_resistance'] > comparative['assistant1_scores']['bias_resistance'] else 'Tie'} |
| Safety Compliance | {comparative['assistant1_scores']['safety_compliance']*100:.1f}% | {comparative['assistant2_scores']['safety_compliance']*100:.1f}% | {'Assistant 1' if comparative['assistant1_scores']['safety_compliance'] > comparative['assistant2_scores']['safety_compliance'] else 'Assistant 2' if comparative['assistant2_scores']['safety_compliance'] > comparative['assistant1_scores']['safety_compliance'] else 'Tie'} |
| **Composite Score** | **{comparative['assistant1_scores']['composite_score']*100:.1f}%** | **{comparative['assistant2_scores']['composite_score']*100:.1f}%** | **{comparative['overall_winner'].replace('assistant', 'Assistant')}** |

### Key Findings

"""

    # Add key findings
    winner = comparative["overall_winner"].replace("assistant", "Assistant")
    gap = comparative["performance_gap"] * 100

    report += f"- **Overall Winner**: {winner} (by {gap:.1f} percentage points)\n"

    # Add recommendations
    report += f"\n### Recommendations\n\n"
    for i, rec in enumerate(results["recommendations"], 1):
        report += f"{i}. {rec}\n"

    report += f"""
## Methodology

This evaluation used:

- **Public Benchmarks**: Curated datasets of factual questions, bias prompts, and safety challenges
- **Custom Prompts**: Domain-specific test cases for comprehensive coverage
- **LLM-as-Judge**: Advanced evaluation using frontier models to score responses
- **Rule-Based Metrics**: Objective measurements of accuracy, bias, and safety violations

The evaluation framework provides reproducible, standardized comparisons across multiple dimensions critical for AI safety and reliability.

## Files Generated

- Detailed JSON results: `demo_results/comprehensive_evaluation_[timestamp].json`
- This summary report: `demo_results/evaluation_summary.md`

---

*Generated by the Comprehensive AI Assistant Evaluation Framework*
"""

    # Save report
    report_path = Path("demo_results/evaluation_summary.md")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"📄 Summary report saved to: {report_path}")


async def main():
    """Main demo function"""

    print("🚀 AI Assistant Evaluation Framework Demo")
    print("=" * 50)
    print()
    print("This demo will compare two AI assistants on:")
    print("  🎯 Hallucination Rate (factual accuracy)")
    print("  ⚖️ Bias & Harmful Outputs (fairness)")
    print("  🛡️ Content Safety (jailbreak resistance)")
    print()

    # Setup environment
    if not setup_demo_environment():
        return 1

    try:
        # Run the evaluation
        print("⏱️ Running evaluation (this may take a few minutes)...")
        results = await run_demo_evaluation()

        # Analyze and display results
        analyze_demo_results(results)

        # Create summary report
        create_demo_summary_report(results)

        print(f"\n🎉 Demo completed successfully!")
        print(f"\n🔍 To explore more:")
        print(f"  📊 View detailed results: demo_results/")
        print(f"  🌐 Launch analysis dashboard: streamlit run evaluation/analysis_dashboard.py")
        print(f"  🔬 Run full evaluation: python evaluation/comprehensive_assistant_evaluation.py")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        print("\nThis might be due to:")
        print("  - Missing or invalid API keys")
        print("  - Network connectivity issues")
        print("  - API rate limits")
        print("\nTry setting up your API keys and running again.")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))