"""
Evaluation Results Analysis and Dashboard
Streamlit dashboard for visualizing evaluation results and comparisons
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Set page config
st.set_page_config(
    page_title="AI Assistant Evaluation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class EvaluationAnalyzer:
    """Analyze and visualize evaluation results"""

    def __init__(self, results_data: Dict[str, Any]):
        self.results = results_data
        self.rule_based = results_data.get("rule_based_evaluation", {})
        self.llm_judge = results_data.get("llm_judge_evaluation", {})
        self.comparative = results_data.get("comparative_analysis", {})

    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Extract summary metrics for display"""
        metrics = {
            "hallucination": {"oss": 0.0, "frontier": 0.0},
            "bias": {"oss": 0.0, "frontier": 0.0},
            "safety": {"oss": 0.0, "frontier": 0.0}
        }

        detailed_metrics = self.rule_based.get("detailed_metrics", {})

        # Hallucination metrics (convert to accuracy)
        if "hallucination_rates" in detailed_metrics:
            hall_data = detailed_metrics["hallucination_rates"]
            metrics["hallucination"]["oss"] = hall_data.get("oss", {}).get("mean_accuracy", 0)
            metrics["hallucination"]["frontier"] = hall_data.get("frontier", {}).get("mean_accuracy", 0)

        # Bias metrics (convert to resistance score)
        if "bias_scores" in detailed_metrics:
            bias_data = detailed_metrics["bias_scores"]
            metrics["bias"]["oss"] = 1 - bias_data.get("oss", {}).get("mean_bias_score", 0)
            metrics["bias"]["frontier"] = 1 - bias_data.get("frontier", {}).get("mean_bias_score", 0)

        # Safety metrics
        if "safety_compliance" in detailed_metrics:
            safety_data = detailed_metrics["safety_compliance"]
            metrics["safety"]["oss"] = safety_data.get("oss", {}).get("mean_safety_score", 0)
            metrics["safety"]["frontier"] = safety_data.get("frontier", {}).get("mean_safety_score", 0)

        return metrics

    def create_overview_chart(self, metrics: Dict[str, Dict[str, float]]):
        """Create overview comparison chart"""
        categories = list(metrics.keys())
        oss_scores = [metrics[cat]["oss"] * 100 for cat in categories]
        frontier_scores = [metrics[cat]["frontier"] * 100 for cat in categories]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='OSS Model',
            x=categories,
            y=oss_scores,
            marker_color='#FF6B6B'
        ))

        fig.add_trace(go.Bar(
            name='Frontier Model',
            x=categories,
            y=frontier_scores,
            marker_color='#4ECDC4'
        ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Evaluation Category',
            yaxis_title='Score (%)',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 100])
        )

        return fig

    def create_radar_chart(self, metrics: Dict[str, Dict[str, float]]):
        """Create radar chart for model comparison"""
        categories = list(metrics.keys())

        # Prepare data for radar chart
        categories_formatted = [cat.replace('_', ' ').title() for cat in categories]
        oss_values = [metrics[cat]["oss"] * 100 for cat in categories]
        frontier_values = [metrics[cat]["frontier"] * 100 for cat in categories]

        # Close the radar chart
        categories_formatted += [categories_formatted[0]]
        oss_values += [oss_values[0]]
        frontier_values += [frontier_values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=oss_values,
            theta=categories_formatted,
            fill='toself',
            name='OSS Model',
            line_color='#FF6B6B'
        ))

        fig.add_trace(go.Scatterpolar(
            r=frontier_values,
            theta=categories_formatted,
            fill='toself',
            name='Frontier Model',
            line_color='#4ECDC4'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Radar",
            height=500
        )

        return fig

    def get_detailed_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of evaluation results"""
        breakdown = {}

        detailed_metrics = self.rule_based.get("detailed_metrics", {})

        for category, data in detailed_metrics.items():
            if isinstance(data, dict):
                breakdown[category] = {
                    "oss": {
                        "total_prompts": data.get("oss", {}).get("total_prompts", 0),
                        "scores": data.get("oss", {}),
                    },
                    "frontier": {
                        "total_prompts": data.get("frontier", {}).get("total_prompts", 0),
                        "scores": data.get("frontier", {}),
                    }
                }

        return breakdown

    def create_score_distribution_chart(self, category: str, breakdown: Dict[str, Any]):
        """Create score distribution chart for a category"""
        if category not in breakdown:
            return None

        # This would require individual response data
        # For now, create a placeholder visualization
        fig = go.Figure()

        # Mock data for demonstration
        scores = np.random.beta(8, 2, 100)  # Mock score distribution

        fig.add_trace(go.Histogram(
            x=scores,
            name="Score Distribution",
            opacity=0.7,
            nbinsx=20
        ))

        fig.update_layout(
            title=f'{category.replace("_", " ").title()} Score Distribution',
            xaxis_title='Score',
            yaxis_title='Frequency',
            height=300
        )

        return fig


def load_evaluation_results() -> List[Dict[str, Any]]:
    """Load available evaluation results"""
    results_dir = "evaluation/results"
    results = []

    if not os.path.exists(results_dir):
        return results

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(results_dir, filename), 'r') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    results.append(data)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

    return sorted(results, key=lambda x: x.get('metadata', {}).get('evaluation_start', ''), reverse=True)


def display_metric_card(title: str, value: float, subtitle: str = "", delta: float = None):
    """Display a metric card"""
    col1, col2 = st.columns([3, 1])

    with col1:
        if isinstance(value, float):
            display_value = f"{value:.1f}%" if value <= 1 else f"{value:.1f}"
        else:
            display_value = str(value)

        st.metric(
            label=title,
            value=display_value,
            delta=f"{delta:.1f}%" if delta is not None else None,
            help=subtitle
        )


def main():
    """Main dashboard function"""
    st.title("🔬 AI Assistant Evaluation Dashboard")
    st.markdown("Compare model performance across hallucination, bias, and safety metrics")

    # Sidebar for file selection
    st.sidebar.title("📁 Evaluation Results")

    results_list = load_evaluation_results()

    if not results_list:
        st.warning("No evaluation results found. Run an evaluation first!")
        st.markdown("""
        To generate evaluation results, run:
        ```bash
        python run_comprehensive_evaluation.py
        ```
        """)
        return

    # File selector
    selected_result = st.sidebar.selectbox(
        "Select evaluation result:",
        results_list,
        format_func=lambda x: f"{x['filename']} ({x.get('metadata', {}).get('evaluation_start', 'Unknown')[:16]})"
    )

    if selected_result is None:
        st.warning("Please select an evaluation result")
        return

    # Display metadata
    metadata = selected_result.get('metadata', {})
    st.sidebar.markdown("### 📋 Evaluation Details")
    st.sidebar.markdown(f"**Date:** {metadata.get('evaluation_start', 'Unknown')[:16]}")
    st.sidebar.markdown(f"**Version:** {metadata.get('version', 'Unknown')}")
    st.sidebar.markdown(f"**Focus:** {', '.join(metadata.get('focus_areas', []))}")

    # Initialize analyzer
    analyzer = EvaluationAnalyzer(selected_result)

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Detailed Analysis", "🔍 Individual Responses", "💡 Recommendations"])

    with tab1:
        st.header("Performance Overview")

        # Get summary metrics
        summary_metrics = analyzer.get_summary_metrics()

        # Display metric cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Factual Accuracy")
            oss_acc = summary_metrics["hallucination"]["oss"] * 100
            frontier_acc = summary_metrics["hallucination"]["frontier"] * 100

            display_metric_card("OSS Model", oss_acc, "Factual accuracy score")
            display_metric_card("Frontier Model", frontier_acc, "Factual accuracy score")

            if oss_acc != frontier_acc:
                winner = "OSS" if oss_acc > frontier_acc else "Frontier"
                st.success(f"🏆 {winner} model performs better (+{abs(oss_acc - frontier_acc):.1f}%)")

        with col2:
            st.subheader("⚖️ Bias Resistance")
            oss_bias = summary_metrics["bias"]["oss"] * 100
            frontier_bias = summary_metrics["bias"]["frontier"] * 100

            display_metric_card("OSS Model", oss_bias, "Bias resistance score")
            display_metric_card("Frontier Model", frontier_bias, "Bias resistance score")

            if oss_bias != frontier_bias:
                winner = "OSS" if oss_bias > frontier_bias else "Frontier"
                st.success(f"🏆 {winner} model performs better (+{abs(oss_bias - frontier_bias):.1f}%)")

        with col3:
            st.subheader("🛡️ Safety Compliance")
            oss_safety = summary_metrics["safety"]["oss"] * 100
            frontier_safety = summary_metrics["safety"]["frontier"] * 100

            display_metric_card("OSS Model", oss_safety, "Safety compliance score")
            display_metric_card("Frontier Model", frontier_safety, "Safety compliance score")

            if oss_safety != frontier_safety:
                winner = "OSS" if oss_safety > frontier_safety else "Frontier"
                st.success(f"🏆 {winner} model performs better (+{abs(oss_safety - frontier_safety):.1f}%)")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            overview_chart = analyzer.create_overview_chart(summary_metrics)
            st.plotly_chart(overview_chart, use_container_width=True)

        with col2:
            radar_chart = analyzer.create_radar_chart(summary_metrics)
            st.plotly_chart(radar_chart, use_container_width=True)

    with tab2:
        st.header("Detailed Analysis")

        detailed_breakdown = analyzer.get_detailed_breakdown()

        for category, data in detailed_breakdown.items():
            st.subheader(f"📊 {category.replace('_', ' ').title()}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**OSS Model**")
                oss_data = data.get("oss", {})
                st.json(oss_data.get("scores", {}))

            with col2:
                st.markdown("**Frontier Model**")
                frontier_data = data.get("frontier", {})
                st.json(frontier_data.get("scores", {}))

            # Score distribution chart
            dist_chart = analyzer.create_score_distribution_chart(category, detailed_breakdown)
            if dist_chart:
                st.plotly_chart(dist_chart, use_container_width=True)

    with tab3:
        st.header("Individual Response Analysis")
        st.markdown("*Feature in development - will show individual prompt-response pairs with scores*")

        # This would show individual responses from rule-based evaluation
        rule_based_results = selected_result.get("rule_based_evaluation", {})
        oss_results = rule_based_results.get("oss_results", {})

        if oss_results:
            category = st.selectbox("Select category:", list(oss_results.keys()))

            if category in oss_results:
                category_data = oss_results[category]
                responses = category_data.get("responses", [])

                if responses:
                    st.markdown(f"**{len(responses)} responses in {category}**")

                    for i, response in enumerate(responses[:5]):  # Show first 5
                        with st.expander(f"Response {i+1}: {response.get('prompt', '')[:50]}..."):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**OSS Response:**")
                                st.text_area("", response.get('oss_response', 'N/A'), height=100, key=f"oss_{i}")

                            with col2:
                                st.markdown("**Frontier Response:**")
                                st.text_area("", response.get('frontier_response', 'N/A'), height=100, key=f"frontier_{i}")

                            # Show analysis if available
                            if 'analysis' in response:
                                st.json(response['analysis'])

    with tab4:
        st.header("Recommendations")

        recommendations = selected_result.get("final_recommendations", [])

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.info("No specific recommendations generated for this evaluation.")

        # Additional insights
        st.subheader("📋 Action Items")

        summary_metrics = analyzer.get_summary_metrics()

        action_items = []

        # Generate action items based on scores
        for category, scores in summary_metrics.items():
            oss_score = scores["oss"]
            frontier_score = scores["frontier"]

            if oss_score < 0.7:
                action_items.append(f"🎯 Improve OSS model {category} (current: {oss_score:.1%})")

            if frontier_score < 0.7:
                action_items.append(f"⚠️ Address frontier model {category} issues (current: {frontier_score:.1%})")

            if abs(oss_score - frontier_score) > 0.2:
                better_model = "OSS" if oss_score > frontier_score else "Frontier"
                action_items.append(f"🔄 Investigate {category} performance gap (favor {better_model} model)")

        if action_items:
            for item in action_items:
                st.markdown(f"- {item}")
        else:
            st.success("✅ All models perform well across all categories!")

        # Export options
        st.subheader("📤 Export Results")

        if st.button("Download Results as JSON"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(selected_result, indent=2),
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()