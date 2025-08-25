# Grok/X-AI Integration with TransparentSF Explainer Agent

## Overview

Grok/X-AI has been successfully integrated into the TransparentSF explainer agent system. You can now use Grok models alongside OpenAI, Anthropic, and Google models.

## Setup

### 1. Install the Integration

The `langchain-xai` package has been added to `requirements.txt` and installed:

```bash
pip install langchain-xai
```

### 2. Get an X-AI API Key

1. Sign up for xAI at [x.ai](https://x.ai)
2. Generate an API key from your account
3. Set the environment variable:

```bash
export XAI_API_KEY="your-api-key-here"
```

Or add it to your `.env` file:

```
XAI_API_KEY=your-api-key-here
```

## Available Grok Models

Two Grok models are now available:

- **`grok-beta`** - The main Grok model with tool calling support
- **`grok-3-latest`** - Latest version with Live Search capabilities

## Usage Examples

### 1. Using Grok with the LangChain Explainer Agent

```python
from ai.agents.langchain_agent.explainer_agent import create_explainer_agent
from ai.agents.langchain_agent.config.tool_config import ToolGroup

# Create an explainer agent with Grok
agent = create_explainer_agent(
    model_key="grok-beta",
    tool_groups=[ToolGroup.CORE, ToolGroup.ANALYSIS],
    enable_session_logging=True
)

# Use the agent
response = agent.explain_change_sync(
    "Explain the recent increase in crime in District 3",
    metric_details={"context": "Crime analysis for District 3"}
)
```

### 2. Using Grok with Live Search

```python
from ai.agents.config.models import create_langchain_llm

# Create a Grok model with Live Search
llm = create_langchain_llm("grok-3-latest")

# The model will automatically use web search for current information
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What are the latest developments in AI regulation?")
]

response = llm.invoke(messages)
```

### 3. Using Grok in the Web Interface

You can now select Grok models in the web interface by setting the `model_key` parameter:

```json
{
    "prompt": "Explain the recent changes in business registrations",
    "model_key": "grok-beta",
    "tool_groups": ["core", "analysis"],
    "session_data": {
        "session_id": "unique_session_id"
    }
}
```

## Model Features

### Grok Beta (`grok-beta`)
- ✅ **Tool calling** - Full support for all explainer agent tools
- ✅ **Structured output** - JSON responses for metric analysis
- ✅ **Token-level streaming** - Real-time responses
- ✅ **Token usage tracking** - Monitor API usage
- ✅ **Logprobs** - Detailed analysis capabilities

### Grok 3 Latest (`grok-3-latest`)
- All features from grok-beta
- ✅ **Live Search** - Real-time web search integration
- Perfect for current events and latest data analysis

## Fallback Support

Grok models are included in the automatic fallback system:

1. `gpt-4o` (OpenAI)
2. `gpt-5` (OpenAI)
3. `grok-beta` (X-AI) ← **New!**
4. `claude-3-5-sonnet` (Anthropic)
5. `grok-3-latest` (X-AI) ← **New!**

## Model Configuration

The system now includes only the latest models:

### OpenAI Models
- `gpt-5` - Latest GPT model (temperature fixed at 1.0)
- `gpt-4o` - GPT-4 Omni (most capable)

### Anthropic Models
- `claude-opus-4` - Latest Claude Opus
- `claude-sonnet-4` - Latest Claude Sonnet
- `claude-3-7-sonnet` - Claude 3.7 Sonnet (default)
- `claude-3-5-sonnet` - Claude 3.5 Sonnet
- `claude-3-5-haiku` - Claude 3.5 Haiku
- `claude-3-haiku` - Claude 3 Haiku

### Google Models
- `gemini-2.5-pro` - Latest Gemini Pro
- `gemini-1.5-pro` - Gemini 1.5 Pro
- `gemini-1.5-flash` - Gemini 1.5 Flash

### Grok Models (X-AI)
- `grok-beta` - Main Grok model with tool calling
- `grok-3-latest` - Latest version with Live Search

## Dynamic Model Loading

All frontend dropdowns now use the `/backend/available-models` API endpoint to dynamically load available models, ensuring:

- Only models with valid API keys are shown
- New models are automatically available when added
- Consistent model selection across all interfaces
- Proper grouping by provider

## Environment Variables

Add these to your `.env` file:

```bash
# X-AI API Key
XAI_API_KEY=your-xai-api-key-here

# Optional: Set Grok as default model
AGENT_MODEL=grok-beta
```

## Testing

To test the integration:

```python
from ai.agents.config.models import get_available_models

# Check available models
available = get_available_models()
print("Available Grok models:", [k for k in available.keys() if k.startswith("grok")])
```

## Benefits of Using Grok

1. **Real-time Information**: Live Search capability for current events
2. **Tool Calling**: Full compatibility with all explainer agent tools
3. **Cost Effective**: Competitive pricing compared to other providers
4. **Reliability**: Included in fallback system for high availability
5. **Streaming**: Real-time responses for better user experience

## Troubleshooting

### Common Issues

1. **"API key not found"**: Make sure `XAI_API_KEY` is set in your environment
2. **Import errors**: Ensure `langchain-xai` is installed in your virtual environment
3. **Tool calling issues**: Verify you're using `grok-beta` or `grok-3-latest` models

### Getting Help

- Check the [LangChain X-AI documentation](https://python.langchain.com/docs/integrations/chat/xai/)
- Review the model configuration in `ai/agents/config/models.py`
- Test with the example code above
