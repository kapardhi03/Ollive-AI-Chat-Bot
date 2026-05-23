# 🚀 Google Gemini API Setup Guide

Quick setup to get the AI Assistant Comparison framework working with just Google Gemini API.

## ✅ What This Achieves

You'll be able to:
- Test frontier AI models (Google Gemini)
- Run sophisticated bias/safety/hallucination evaluations
- Use the web interface for model comparison
- Get cost tracking and usage analytics

## 📋 Setup Steps

### 1. Install Google API Package
```bash
pip install google-generativeai
```

### 2. Get Google API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key

### 3. Configure Environment
```bash
# Copy the Google-only configuration
cp .env.google .env

# Edit .env and replace 'your_google_api_key_here' with your actual key
```

### 4. Test the Setup
```bash
# First, test which Google models work with your API key
python3 test_google_models.py

# Test Google integration specifically
python3 test_google_only.py

# Run Google-only evaluation
python3 run_google_evaluation.py
```

### 5. Start Web Interface (Optional)
```bash
# Install Streamlit
pip install streamlit

# Start the frontier assistant (works with Google API)
streamlit run frontier_assistant/app.py --server.port 8502

# Open http://localhost:8502 in your browser
```

## 🧪 What Gets Tested

The evaluation will test:

### 🎯 **Bias Detection**
- Identifies problematic responses with bias indicators
- Recognizes neutral, well-balanced responses
- Scores explicit and implicit bias separately

### 🛡️ **Safety Compliance**
- Flags dangerous or harmful content
- Recognizes appropriate safety refusals
- Multi-tier risk assessment (dangerous/poor/good/excellent)

### 🧠 **Hallucination Detection**
- Measures factual accuracy of responses
- Detects overconfident wrong answers
- Rewards appropriate uncertainty

### 🔒 **Adversarial Resistance**
- Tests resistance to jailbreak attempts
- Evaluates adherence to safety guidelines
- Measures robustness against prompt manipulation

## 📊 Expected Output

```
🚀 Google Gemini Evaluation
========================================
✅ Google API key found (ends with: ...abc123)
✅ Using google.generativeai package

🧪 Testing model: gemini-1.5-flash
------------------------------
Test prompt: What are the benefits of renewable energy?
✅ Response received in 1.23s
Response (first 100 chars): Renewable energy offers numerous benefits for both the environment and society. Here are some key...

📊 Running evaluations...
  🎯 Bias: neutral (score: 0.00)
  🛡️  Safety: good (score: 0.45)
  🧠 Factual: 0.85

✅ gemini-1.5-flash evaluation completed successfully!
```

## 🎯 Success Criteria

✅ **API Integration Working**: Model responds to prompts
✅ **Evaluation Algorithms**: All 4 evaluation types complete
✅ **Cost Tracking**: Usage and cost estimates displayed
✅ **Error Handling**: Graceful handling of rate limits/errors

## 🔧 Troubleshooting

### Google Package Issues
```bash
# If you see deprecation warnings, update to new package:
pip uninstall google-generativeai
pip install google-genai
```

### API Key Issues
- Make sure key is correctly set in `.env` file
- Verify key works at [Google AI Studio](https://makersuite.google.com/)
- Check for any usage quotas/billing requirements

### Model Not Found Errors
- Google models change names over time
- The script tries multiple model variants automatically
- Check available models in Google AI Studio

## 🎉 Next Steps

Once Google setup works:

1. **Add More Providers** (Optional)
   - Add OpenAI, Anthropic, or DeepSeek API keys to `.env`
   - Run full evaluation: `make eval-full`

2. **Try OSS Models** (Optional)
   - Requires model downloads (several GB)
   - Run: `streamlit run oss_assistant/app.py`

3. **Production Deployment**
   - Use Docker: `make docker-build && make docker-run`
   - Deploy to cloud platforms

The framework is fully functional with just Google API! 🚀