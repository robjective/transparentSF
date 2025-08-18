# Enhanced Session Logging for LangChain Explainer Agent

## Overview

The LangChain Explainer Agent now includes comprehensive session logging that captures detailed information about each interaction for debugging and improvement purposes.

## Features

### Session Logging Structure

Each session is logged as a structured JSON file containing:

- **Session Metadata**: ID, timestamp, model, configuration
- **User Input**: The original prompt/query
- **Conversation History**: Complete message exchange
- **Tool Calls**: Detailed information about each tool execution
- **Performance Metrics**: Execution times and success rates
- **Error Information**: Detailed error tracking and categorization

### Tool Call Tracking

For each tool call, the system logs:

- **Tool Name**: The specific tool being called
- **Arguments**: Input parameters (sanitized for JSON serialization)
- **Timing**: Start time, end time, and execution duration
- **Results**: Output data or error messages
- **Success Status**: Whether the tool call succeeded or failed
- **Error Analysis**: Automatic detection of common error patterns

### Error Detection

The system automatically detects and categorizes errors:

- **API Errors**: Network timeouts, authentication failures
- **Validation Errors**: Invalid parameters, missing required fields
- **Business Logic Errors**: Data not found, permission denied
- **System Errors**: Unexpected exceptions, parsing failures

## Configuration

### Enabling Session Logging

```python
from agents.langchain_agent.explainer_agent import LangChainExplainerAgent
from agents.langchain_agent.config.tool_config import ToolGroup

# Create agent with session logging enabled (default)
agent = LangChainExplainerAgent(
    model_key="gpt-4o",
    tool_groups=[ToolGroup.CORE],
    enable_session_logging=True  # Default is True
)

# Disable session logging if needed
agent = LangChainExplainerAgent(
    model_key="gpt-4o",
    tool_groups=[ToolGroup.CORE],
    enable_session_logging=False
)
```

### Log File Location

Session logs are stored in: `ai/logs/sessions/`

File naming convention: `session_YYYYMMDD_HHMMSS_sessionid.json`

## Log File Structure

```json
{
  "session_id": "uuid-string",
  "timestamp": "2025-01-15T10:30:00Z",
  "agent_type": "langchain_explainer",
  "model": "gpt-4o",
  "user_input": "Explain the crime increase in district 3",
  "conversation": [
    {
      "role": "user",
      "content": "Explain the crime increase in district 3",
      "timestamp": "2025-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "I'll help you analyze the crime data...",
      "timestamp": "2025-01-15T10:30:15Z",
      "sender": "LangChain Explainer"
    }
  ],
  "tool_calls": [
    {
      "tool_name": "get_dashboard_metric",
      "arguments": {
        "district_number": 3,
        "metric_id": "crime_incidents"
      },
      "start_time": 1705312200.123,
      "end_time": 1705312202.456,
      "success": true,
      "result": {
        "data": "...",
        "status": "success"
      },
      "error_message": null,
      "execution_time_ms": 2333
    }
  ],
  "final_response": "Based on the data analysis...",
  "total_execution_time_ms": 15000,
  "success": true,
  "error_summary": null,
  "model_config": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "tool_groups": ["CORE"],
  "available_tools": ["get_dashboard_metric", "query_anomalies_db", ...]
}
```

## Usage Examples

### Basic Usage

```python
# Create agent with session logging
agent = LangChainExplainerAgent(enable_session_logging=True)

# Run a query
result = agent.explain_change_sync(
    "Explain the recent changes in housing permits",
    metric_details={}
)

# Check session ID in result
session_id = result.get('session_id')
print(f"Session logged with ID: {session_id}")
```

### Analyzing Session Logs

```python
import json
from pathlib import Path

# Find session log by ID
logs_dir = Path("ai/logs/sessions")
session_files = list(logs_dir.glob(f"*{session_id[:8]}*.json"))

if session_files:
    with open(session_files[0], 'r') as f:
        session_data = json.load(f)
    
    # Analyze tool calls
    tool_calls = session_data['tool_calls']
    successful_calls = [tc for tc in tool_calls if tc['success']]
    failed_calls = [tc for tc in tool_calls if not tc['success']]
    
    print(f"Total tool calls: {len(tool_calls)}")
    print(f"Successful: {len(successful_calls)}")
    print(f"Failed: {len(failed_calls)}")
    
    # Show failed tool calls
    for tc in failed_calls:
        print(f"Failed: {tc['tool_name']} - {tc['error_message']}")
```

## Benefits

### For Development

1. **Debugging**: Complete visibility into agent behavior
2. **Performance Analysis**: Track execution times and bottlenecks
3. **Error Investigation**: Detailed error context and categorization
4. **Tool Validation**: Verify tool calls and parameter passing

### For Improvement

1. **Pattern Analysis**: Identify common failure modes
2. **Performance Optimization**: Find slow operations
3. **Prompt Engineering**: Analyze conversation flows
4. **Tool Selection**: Understand which tools are most/least effective

### For Monitoring

1. **Success Rates**: Track overall agent performance
2. **Error Trends**: Monitor for recurring issues
3. **Usage Patterns**: Understand user behavior
4. **Resource Usage**: Monitor execution times and costs

## Testing

Run the test script to verify session logging:

```bash
cd ai/agents/langchain_agent
python test_session_logging.py
```

This will:
1. Create an agent with session logging enabled
2. Execute a test query
3. Verify the session log was created
4. Display a summary of the logged data

## Dependencies

The session logging feature requires:

- `dataclasses-json>=0.6.0` (added to requirements.txt)
- Standard Python libraries (json, datetime, pathlib, etc.)

## Future Enhancements

Potential improvements for future versions:

1. **Database Integration**: Store logs in PostgreSQL for querying
2. **Real-time Monitoring**: Live dashboard for session tracking
3. **Advanced Analytics**: ML-based pattern detection
4. **Log Rotation**: Automatic cleanup of old logs
5. **Compression**: Reduce storage requirements
6. **Search Indexing**: Full-text search across logs
