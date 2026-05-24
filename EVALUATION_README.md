# AI Assistant Evaluation Framework

A comprehensive evaluation framework for comparing AI assistants on **hallucination rate**, **bias & harmful outputs**, and **content safety**. This framework implements both rule-based metrics and LLM-as-judge approaches for rigorous, reproducible evaluations.

## 🎯 Evaluation Dimensions

### 1. Hallucination Rate
- **Factual Accuracy**: Tests responses against known facts
- **Confidence Calibration**: Evaluates appropriateness of confidence levels
- **Source Attribution**: Checks for fabricated citations or studies
- **Uncertainty Handling**: Assesses appropriate expressions of uncertainty

### 2. Bias & Harmful Outputs
- **Stereotyping**: Detection of harmful stereotypes across demographics
- **Discriminatory Content**: Identification of unfair treatment recommendations
- **Balanced Perspectives**: Evaluation of nuanced, fair viewpoints
- **Appropriate Refusals**: Assessment of declining problematic requests

### 3. Content Safety
- **Jailbreak Resistance**: Testing against various attack vectors
- **Harmful Content Generation**: Detection of dangerous instructions
- **Safety Compliance**: Adherence to content policies
- **Refusal Appropriateness**: Quality of safety-related refusals

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt

# Set up API keys (choose one or more)
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Run the Demo

```bash
# Quick demonstration comparing GPT-4 vs Claude
python demo_evaluation_comparison.py
```

### Full Evaluation

```bash
# Comprehensive evaluation
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 anthropic:claude-3-sonnet-20240229 \
  --mode comprehensive

# Quick evaluation (fewer test cases)
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 anthropic:claude-3-sonnet-20240229 \
  --mode quick

# Without LLM-as-judge (faster, rule-based only)
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 anthropic:claude-3-sonnet-20240229 \
  --no-llm-judge
```

## 📊 Analysis & Visualization

### Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run evaluation/analysis_dashboard.py
```

The dashboard provides:
- **Performance Overview**: Comparative scores across all dimensions
- **Detailed Analysis**: Category-specific breakdowns and metrics
- **Individual Responses**: Examination of specific prompt-response pairs
- **Recommendations**: Actionable insights for improvement

### Results Structure

Evaluation results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "evaluation_timestamp": "2024-01-15T10:30:00",
    "framework_version": "2.0.0",
    "evaluation_focus": ["hallucination_rate", "bias_and_harmful_outputs", "content_safety"]
  },
  "assistant_configs": {
    "assistant1": {"provider": "openai", "model": "gpt-4-turbo-preview"},
    "assistant2": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"}
  },
  "evaluation_results": {
    "hallucination_assessment": { /* detailed factual accuracy results */ },
    "bias_assessment": { /* bias and harmful content analysis */ },
    "safety_assessment": { /* safety compliance and jailbreak resistance */ },
    "comparative_scores": {
      "assistant1_scores": {
        "factual_accuracy": 0.85,
        "bias_resistance": 0.92,
        "safety_compliance": 0.88,
        "composite_score": 0.88
      },
      "assistant2_scores": {
        "factual_accuracy": 0.89,
        "bias_resistance": 0.87,
        "safety_compliance": 0.91,
        "composite_score": 0.89
      },
      "overall_winner": "assistant2"
    }
  },
  "recommendations": [ /* actionable improvement suggestions */ ]
}
```

## 🧪 Evaluation Methodologies

### Rule-Based Metrics

1. **Factual Accuracy Scoring**
   - Exact answer matching
   - Keyword presence analysis
   - Confidence/uncertainty detection
   - Fabrication indicators

2. **Bias Detection**
   - Stereotype keyword matching
   - Balanced perspective indicators
   - Appropriate refusal patterns
   - Harmful content flags

3. **Safety Assessment**
   - Refusal quality scoring
   - Harmful instruction detection
   - Jailbreak success identification
   - Policy compliance checking

### LLM-as-Judge Evaluation

Uses advanced language models to provide nuanced scoring on:
- Response appropriateness
- Content quality and helpfulness
- Subtle bias detection
- Safety compliance assessment
- Comparative ranking

### Public Benchmarks Integrated

- **Factual Knowledge**: Geography, history, science, current events
- **Bias Prompts**: Gender, racial, religious, socioeconomic scenarios
- **Safety Challenges**: Jailbreak attempts, harmful requests, edge cases
- **Adversarial Prompts**: Sophisticated attack vectors and manipulation attempts

## 📁 Project Structure

```
evaluation/
├── comprehensive_assistant_evaluation.py    # Main evaluation script
├── enhanced_benchmark.py                    # Core evaluation logic
├── llm_judge.py                            # LLM-as-judge implementation
├── analysis_dashboard.py                   # Streamlit visualization
├── datasets/
│   └── comprehensive_prompts.json          # Test datasets
└── results/                                # Generated evaluation results

demo_evaluation_comparison.py               # Quick demo script
EVALUATION_README.md                        # This file
```

## 🔧 Customization

### Adding Custom Test Cases

Extend the evaluation by modifying `evaluation/datasets/comprehensive_prompts.json`:

```json
{
  "custom_category": {
    "description": "Custom evaluation category",
    "prompts": [
      {
        "prompt": "Your custom test prompt",
        "expected_behavior": "refuse",
        "category": "custom",
        "risk_level": "medium"
      }
    ]
  }
}
```

### Custom Evaluation Criteria

Extend the `ComprehensiveAssistantEvaluator` class:

```python
def _evaluate_custom_dimension(self, client1, client2, config1, config2, prompts):
    """Add your custom evaluation dimension"""
    # Your custom evaluation logic here
    pass
```

### Custom LLM Judge Templates

Modify `evaluation/llm_judge.py` to add custom evaluation templates:

```python
def _get_custom_template(self) -> str:
    return """
    Evaluate the response for your custom criteria:

    Prompt: {prompt}
    Response: {response}

    Please evaluate on:
    1. Custom Criterion 1 (0-10)
    2. Custom Criterion 2 (0-10)

    Provide JSON format: {{"criterion1": X, "criterion2": X}}
    """
```

## 📈 Interpreting Results

### Score Interpretation

- **Factual Accuracy**: 0.0-1.0 (higher = more accurate, less hallucination)
- **Bias Resistance**: 0.0-1.0 (higher = less biased, more fair)
- **Safety Compliance**: 0.0-1.0 (higher = safer, better refusals)
- **Composite Score**: Weighted average of all dimensions

### Performance Thresholds

- **Excellent**: > 0.9 (>90%)
- **Good**: 0.8-0.9 (80-90%)
- **Adequate**: 0.6-0.8 (60-80%)
- **Concerning**: 0.4-0.6 (40-60%)
- **Poor**: < 0.4 (<40%)

### Red Flags

Watch for these concerning patterns:
- High hallucination rate (>20%)
- Frequent biased responses (>30% problematic)
- Safety violations (any jailbreak successes)
- Inconsistent behavior across similar prompts

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. **Rate Limiting**
   - Reduce evaluation mode to "quick"
   - Add delays between API calls
   - Use multiple API keys if available

3. **Missing Dependencies**
   ```bash
   pip install streamlit plotly pandas anthropic openai
   ```

4. **Memory Issues**
   - Use "quick" mode for large evaluations
   - Process results in batches
   - Clear cache between runs

### Performance Optimization

- **Parallel Processing**: Enable concurrent API calls
- **Caching**: Store intermediate results
- **Batch Evaluation**: Group similar prompts
- **Smart Sampling**: Use representative subsets for large datasets

## 📊 Example Use Cases

### 1. Model Selection
Compare multiple models to choose the best for your use case:

```bash
# Compare three models
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 anthropic:claude-3-sonnet-20240229

python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 google:gemini-pro
```

### 2. Fine-Tuning Validation
Evaluate improvements after fine-tuning:

```bash
# Before fine-tuning
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 openai:gpt-4-base

# After fine-tuning
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 openai:gpt-4-turbo-preview \
  --assistant2 your-org:custom-fine-tuned-model
```

### 3. Safety Auditing
Focus on safety-critical applications:

```bash
# Run comprehensive safety evaluation
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 your-production-model \
  --assistant2 safety-baseline-model \
  --mode comprehensive
```

### 4. Bias Assessment
Evaluate fairness across demographics:

```bash
# Focus on bias evaluation with LLM judge
python evaluation/comprehensive_assistant_evaluation.py \
  --assistant1 candidate-model \
  --assistant2 bias-tested-baseline \
  --mode comprehensive
```

## 🤝 Contributing

We welcome contributions! Please:

1. **Add Test Cases**: Expand the evaluation datasets
2. **Improve Metrics**: Enhance scoring algorithms
3. **Add Visualizations**: Create new dashboard components
4. **Optimize Performance**: Speed up evaluation processes

### Development Setup

```bash
git clone [repository]
cd ai-assistant-evaluation
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 evaluation/
black evaluation/
```

## 📝 Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{ai_assistant_evaluation_2024,
  title={Comprehensive AI Assistant Evaluation Framework},
  author={[Your Name]},
  year={2024},
  url={[repository_url]},
  note={Framework for evaluating hallucination, bias, and safety in AI assistants}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Evaluation methodologies inspired by academic research in AI safety
- Test datasets curated from public benchmarks and original content
- LLM-as-judge approaches based on recent advances in model-based evaluation
- Community contributions to bias detection and safety assessment techniques

---

**Note**: This evaluation framework is designed for research and development purposes. Always supplement automated evaluations with human review for production systems.