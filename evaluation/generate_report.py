"""
Evaluation Report Generator
Generate comprehensive PDF and HTML reports from evaluation results
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO


class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports"""

    def __init__(self, results_data: Dict[str, Any]):
        self.results = results_data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_html_report(self, output_path: str = None) -> str:
        """Generate HTML report"""
        if output_path is None:
            output_path = f"evaluation_report_{self.timestamp}.html"

        # Extract key metrics
        metrics = self._extract_metrics()
        charts = self._generate_charts()

        # Load template
        template_html = self._get_html_template()
        template = Template(template_html)

        # Render template
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=self.results,
            metrics=metrics,
            charts=charts,
            recommendations=self.results.get("final_recommendations", [])
        )

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def generate_markdown_summary(self, output_path: str = None) -> str:
        """Generate markdown summary report"""
        if output_path is None:
            output_path = f"evaluation_summary_{self.timestamp}.md"

        metrics = self._extract_metrics()
        markdown_content = self._generate_markdown_content(metrics)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return output_path

    def _extract_metrics(self) -> Dict[str, Any]:
        """Extract key metrics from results"""
        metrics = {
            "hallucination": {"oss": 0.0, "frontier": 0.0},
            "bias": {"oss": 0.0, "frontier": 0.0},
            "safety": {"oss": 0.0, "frontier": 0.0},
            "overall": {"oss": 0.0, "frontier": 0.0}
        }

        detailed_metrics = self.results.get("rule_based_evaluation", {}).get("detailed_metrics", {})

        # Extract hallucination metrics
        if "hallucination_rates" in detailed_metrics:
            hall_data = detailed_metrics["hallucination_rates"]
            metrics["hallucination"]["oss"] = hall_data.get("oss", {}).get("mean_accuracy", 0)
            metrics["hallucination"]["frontier"] = hall_data.get("frontier", {}).get("mean_accuracy", 0)

        # Extract bias metrics
        if "bias_scores" in detailed_metrics:
            bias_data = detailed_metrics["bias_scores"]
            metrics["bias"]["oss"] = 1 - bias_data.get("oss", {}).get("mean_bias_score", 0)
            metrics["bias"]["frontier"] = 1 - bias_data.get("frontier", {}).get("mean_bias_score", 0)

        # Extract safety metrics
        if "safety_compliance" in detailed_metrics:
            safety_data = detailed_metrics["safety_compliance"]
            metrics["safety"]["oss"] = safety_data.get("oss", {}).get("mean_safety_score", 0)
            metrics["safety"]["frontier"] = safety_data.get("frontier", {}).get("mean_safety_score", 0)

        # Calculate overall scores
        categories = ["hallucination", "bias", "safety"]
        for model in ["oss", "frontier"]:
            scores = [metrics[cat][model] for cat in categories if metrics[cat][model] > 0]
            metrics["overall"][model] = sum(scores) / len(scores) if scores else 0

        return metrics

    def _generate_charts(self) -> Dict[str, str]:
        """Generate charts as base64 encoded images"""
        charts = {}

        metrics = self._extract_metrics()

        # Overview comparison chart
        overview_fig = self._create_overview_chart(metrics)
        charts["overview"] = self._fig_to_base64(overview_fig)

        # Radar chart
        radar_fig = self._create_radar_chart(metrics)
        charts["radar"] = self._fig_to_base64(radar_fig)

        # Category breakdown
        breakdown_fig = self._create_breakdown_chart(metrics)
        charts["breakdown"] = self._fig_to_base64(breakdown_fig)

        return charts

    def _create_overview_chart(self, metrics: Dict[str, Dict[str, float]]):
        """Create overview comparison chart"""
        categories = ["hallucination", "bias", "safety"]
        oss_scores = [metrics[cat]["oss"] * 100 for cat in categories]
        frontier_scores = [metrics[cat]["frontier"] * 100 for cat in categories]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='OSS Model',
            x=categories,
            y=oss_scores,
            marker_color='#FF6B6B',
            text=[f'{score:.1f}%' for score in oss_scores],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='Frontier Model',
            x=categories,
            y=frontier_scores,
            marker_color='#4ECDC4',
            text=[f'{score:.1f}%' for score in frontier_scores],
            textposition='auto'
        ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Evaluation Category',
            yaxis_title='Score (%)',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 100]),
            font=dict(size=12),
            template='plotly_white'
        )

        return fig

    def _create_radar_chart(self, metrics: Dict[str, Dict[str, float]]):
        """Create radar chart for model comparison"""
        categories = ["hallucination", "bias", "safety"]
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
            line_color='#FF6B6B',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))

        fig.add_trace(go.Scatterpolar(
            r=frontier_values,
            theta=categories_formatted,
            fill='toself',
            name='Frontier Model',
            line_color='#4ECDC4',
            fillcolor='rgba(78, 205, 196, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Radar",
            height=500,
            font=dict(size=12),
            template='plotly_white'
        )

        return fig

    def _create_breakdown_chart(self, metrics: Dict[str, Dict[str, float]]):
        """Create detailed breakdown chart"""
        detailed_metrics = self.results.get("rule_based_evaluation", {}).get("detailed_metrics", {})

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Hallucination Rate', 'Bias Score', 'Safety Compliance'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )

        # Hallucination data
        if "hallucination_rates" in detailed_metrics:
            hall_data = detailed_metrics["hallucination_rates"]
            oss_rate = (1 - hall_data.get("oss", {}).get("mean_accuracy", 0)) * 100
            frontier_rate = (1 - hall_data.get("frontier", {}).get("mean_accuracy", 0)) * 100

            fig.add_trace(
                go.Bar(x=['OSS', 'Frontier'], y=[oss_rate, frontier_rate],
                      name='Hallucination Rate', marker_color=['#FF6B6B', '#4ECDC4'],
                      showlegend=False),
                row=1, col=1
            )

        # Bias data
        if "bias_scores" in detailed_metrics:
            bias_data = detailed_metrics["bias_scores"]
            oss_bias = bias_data.get("oss", {}).get("mean_bias_score", 0) * 100
            frontier_bias = bias_data.get("frontier", {}).get("mean_bias_score", 0) * 100

            fig.add_trace(
                go.Bar(x=['OSS', 'Frontier'], y=[oss_bias, frontier_bias],
                      name='Bias Score', marker_color=['#FF6B6B', '#4ECDC4'],
                      showlegend=False),
                row=1, col=2
            )

        # Safety data
        if "safety_compliance" in detailed_metrics:
            safety_data = detailed_metrics["safety_compliance"]
            oss_safety = safety_data.get("oss", {}).get("mean_safety_score", 0) * 100
            frontier_safety = safety_data.get("frontier", {}).get("mean_safety_score", 0) * 100

            fig.add_trace(
                go.Bar(x=['OSS', 'Frontier'], y=[oss_safety, frontier_safety],
                      name='Safety Score', marker_color=['#FF6B6B', '#4ECDC4'],
                      showlegend=False),
                row=1, col=3
            )

        fig.update_layout(height=400, template='plotly_white')

        return fig

    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        buffer = BytesIO()
        fig.write_image(buffer, format='png', width=800, height=600, scale=2)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    def _generate_markdown_content(self, metrics: Dict[str, Any]) -> str:
        """Generate markdown content"""
        metadata = self.results.get("metadata", {})

        content = f"""# AI Assistant Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Evaluation Version:** {metadata.get('version', 'Unknown')}
**Focus Areas:** {', '.join(metadata.get('focus_areas', []))}

## Executive Summary

This report compares the performance of an OSS (Open Source Software) model against a frontier model across three critical evaluation dimensions:

1. **Hallucination Rate**: Tendency to generate false or unverifiable information
2. **Bias & Harmful Outputs**: Presence of biased, stereotypical, or discriminatory content
3. **Content Safety**: Resistance to jailbreak attempts and harmful content generation

## Performance Overview

### Overall Scores

| Model | Factual Accuracy | Bias Resistance | Safety Compliance | Overall |
|-------|------------------|-----------------|-------------------|---------|
| OSS Model | {metrics['hallucination']['oss']:.1%} | {metrics['bias']['oss']:.1%} | {metrics['safety']['oss']:.1%} | {metrics['overall']['oss']:.1%} |
| Frontier Model | {metrics['hallucination']['frontier']:.1%} | {metrics['bias']['frontier']:.1%} | {metrics['safety']['frontier']:.1%} | {metrics['overall']['frontier']:.1%} |

### Key Findings

"""

        # Determine winners
        better_accuracy = "OSS" if metrics['hallucination']['oss'] > metrics['hallucination']['frontier'] else "Frontier"
        better_bias = "OSS" if metrics['bias']['oss'] > metrics['bias']['frontier'] else "Frontier"
        better_safety = "OSS" if metrics['safety']['oss'] > metrics['safety']['frontier'] else "Frontier"
        better_overall = "OSS" if metrics['overall']['oss'] > metrics['overall']['frontier'] else "Frontier"

        content += f"""
- **Factual Accuracy Winner**: {better_accuracy} Model
- **Bias Resistance Winner**: {better_bias} Model
- **Safety Compliance Winner**: {better_safety} Model
- **Overall Winner**: {better_overall} Model

## Detailed Analysis

### 🎯 Factual Accuracy & Hallucination

"""

        detailed_metrics = self.results.get("rule_based_evaluation", {}).get("detailed_metrics", {})
        if "hallucination_rates" in detailed_metrics:
            hall_data = detailed_metrics["hallucination_rates"]
            oss_hall = hall_data.get("oss", {})
            frontier_hall = hall_data.get("frontier", {})

            content += f"""
- **OSS Model**: {oss_hall.get('hallucination_rate', 0):.1%} hallucination rate ({oss_hall.get('total_prompts', 0)} prompts tested)
- **Frontier Model**: {frontier_hall.get('hallucination_rate', 0):.1%} hallucination rate ({frontier_hall.get('total_prompts', 0)} prompts tested)

"""

        content += """### ⚖️ Bias & Harmful Content

"""

        if "bias_scores" in detailed_metrics:
            bias_data = detailed_metrics["bias_scores"]
            oss_bias = bias_data.get("oss", {})
            frontier_bias = bias_data.get("frontier", {})

            content += f"""
- **OSS Model**: {oss_bias.get('mean_bias_score', 0):.1%} average bias score, {oss_bias.get('high_risk_responses', 0)} high-risk responses
- **Frontier Model**: {frontier_bias.get('mean_bias_score', 0):.1%} average bias score, {frontier_bias.get('high_risk_responses', 0)} high-risk responses

"""

        content += """### 🛡️ Safety & Jailbreak Resistance

"""

        if "safety_compliance" in detailed_metrics:
            safety_data = detailed_metrics["safety_compliance"]
            oss_safety = safety_data.get("oss", {})
            frontier_safety = safety_data.get("frontier", {})

            content += f"""
- **OSS Model**: {oss_safety.get('mean_safety_score', 0):.1%} safety compliance, {oss_safety.get('dangerous_responses', 0)} dangerous responses
- **Frontier Model**: {frontier_safety.get('mean_safety_score', 0):.1%} safety compliance, {frontier_safety.get('dangerous_responses', 0)} dangerous responses

"""

        # Recommendations
        recommendations = self.results.get("final_recommendations", [])
        if recommendations:
            content += """## Recommendations

"""
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"

        # Methodology
        content += """
## Methodology

This evaluation employed a dual approach:

1. **Rule-based Evaluation**: Automated analysis using predefined patterns and scoring algorithms
2. **LLM-as-Judge Evaluation**: Advanced evaluation using frontier models to assess response quality

### Test Categories

- **Factual Knowledge**: 20 questions with known answers to test for hallucinations
- **Bias Prompts**: 20 prompts designed to elicit biased or stereotypical responses
- **Safety Prompts**: 20 adversarial prompts testing jailbreak resistance and harmful content generation

### Scoring

All scores are normalized to 0-100% scale where higher scores indicate better performance:
- **Factual Accuracy**: Based on correctness and appropriate confidence levels
- **Bias Resistance**: Inverse of bias score (lower bias = higher resistance)
- **Safety Compliance**: Based on refusal appropriateness and absence of harmful content

---

*This report was generated automatically by the AI Assistant Evaluation Framework*
"""

        return content

    def _get_html_template(self) -> str:
        """Get HTML template for report"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .metric-card { background: #f8f9fa; padding: 20px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 200px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 0.9em; }
        .chart { margin: 20px 0; text-align: center; }
        .recommendations { background: #e8f4fd; padding: 20px; border-left: 4px solid #2196F3; margin: 20px 0; }
        .winner { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
        th { background-color: #f2f2f2; }
        .oss { color: #FF6B6B; font-weight: bold; }
        .frontier { color: #4ECDC4; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 AI Assistant Evaluation Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Evaluation Version: {{ results.metadata.version or 'Unknown' }}</p>
    </div>

    <section>
        <h2>📊 Performance Overview</h2>

        <div class="metric-card">
            <div class="metric-label">OSS Model - Overall Score</div>
            <div class="metric-value oss">{{ "%.1f"|format(metrics.overall.oss * 100) }}%</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Frontier Model - Overall Score</div>
            <div class="metric-value frontier">{{ "%.1f"|format(metrics.overall.frontier * 100) }}%</div>
        </div>

        {% if metrics.overall.oss > metrics.overall.frontier %}
        <div class="winner">🏆 OSS Model performs better overall (+{{ "%.1f"|format((metrics.overall.oss - metrics.overall.frontier) * 100) }}%)</div>
        {% elif metrics.overall.frontier > metrics.overall.oss %}
        <div class="winner">🏆 Frontier Model performs better overall (+{{ "%.1f"|format((metrics.overall.frontier - metrics.overall.oss) * 100) }}%)</div>
        {% else %}
        <div class="winner">🤝 Models perform equally overall</div>
        {% endif %}

        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>OSS Model</th>
                    <th>Frontier Model</th>
                    <th>Winner</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>🎯 Factual Accuracy</td>
                    <td class="oss">{{ "%.1f"|format(metrics.hallucination.oss * 100) }}%</td>
                    <td class="frontier">{{ "%.1f"|format(metrics.hallucination.frontier * 100) }}%</td>
                    <td>
                        {% if metrics.hallucination.oss > metrics.hallucination.frontier %}OSS{%
                        elif metrics.hallucination.frontier > metrics.hallucination.oss %}Frontier{%
                        else %}Tie{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>⚖️ Bias Resistance</td>
                    <td class="oss">{{ "%.1f"|format(metrics.bias.oss * 100) }}%</td>
                    <td class="frontier">{{ "%.1f"|format(metrics.bias.frontier * 100) }}%</td>
                    <td>
                        {% if metrics.bias.oss > metrics.bias.frontier %}OSS{%
                        elif metrics.bias.frontier > metrics.bias.oss %}Frontier{%
                        else %}Tie{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>🛡️ Safety Compliance</td>
                    <td class="oss">{{ "%.1f"|format(metrics.safety.oss * 100) }}%</td>
                    <td class="frontier">{{ "%.1f"|format(metrics.safety.frontier * 100) }}%</td>
                    <td>
                        {% if metrics.safety.oss > metrics.safety.frontier %}OSS{%
                        elif metrics.safety.frontier > metrics.safety.oss %}Frontier{%
                        else %}Tie{% endif %}
                    </td>
                </tr>
            </tbody>
        </table>
    </section>

    <section>
        <h2>📈 Visual Analysis</h2>

        <div class="chart">
            <h3>Performance Comparison</h3>
            <img src="data:image/png;base64,{{ charts.overview }}" alt="Performance Overview" style="max-width: 100%; height: auto;">
        </div>

        <div class="chart">
            <h3>Model Performance Radar</h3>
            <img src="data:image/png;base64,{{ charts.radar }}" alt="Performance Radar" style="max-width: 100%; height: auto;">
        </div>
    </section>

    {% if recommendations %}
    <section class="recommendations">
        <h2>💡 Recommendations</h2>
        <ol>
            {% for rec in recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ol>
    </section>
    {% endif %}

    <section>
        <h2>📋 Methodology</h2>
        <p>This evaluation employed a comprehensive approach combining rule-based analysis with LLM-as-judge evaluation across three key dimensions:</p>

        <ul>
            <li><strong>Hallucination Rate:</strong> Factual accuracy testing with known answers</li>
            <li><strong>Bias & Harmful Outputs:</strong> Bias detection and stereotype assessment</li>
            <li><strong>Content Safety:</strong> Jailbreak resistance and harmful content filtering</li>
        </ul>

        <p>All metrics are normalized to a 0-100% scale for consistent comparison.</p>
    </section>

    <footer style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>Generated by AI Assistant Evaluation Framework</p>
    </footer>
</body>
</html>
"""


def main():
    """Main function for report generation"""
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("results_file", help="Path to evaluation results JSON file")
    parser.add_argument("--format", choices=["html", "markdown", "both"],
                       default="both", help="Report format")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    # Load results
    try:
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load results file: {e}")
        return 1

    # Generate reports
    generator = EvaluationReportGenerator(results_data)

    try:
        if args.format in ["html", "both"]:
            html_path = os.path.join(args.output_dir, f"evaluation_report_{generator.timestamp}.html")
            html_file = generator.generate_html_report(html_path)
            print(f"✅ HTML report generated: {html_file}")

        if args.format in ["markdown", "both"]:
            md_path = os.path.join(args.output_dir, f"evaluation_summary_{generator.timestamp}.md")
            md_file = generator.generate_markdown_summary(md_path)
            print(f"✅ Markdown report generated: {md_file}")

        print("\n🎉 Report generation completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())