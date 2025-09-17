# LangSmith Integration Setup Guide

## 🔍 Overview

LangSmith is now fully integrated into AroBot to provide comprehensive monitoring and observability for the entire LLM-as-Agent workflow. This guide explains how to set up and use LangSmith tracing.

## 🚀 Quick Setup

### 1. Get Your LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account or sign in
3. Generate an API key from your settings

### 2. Configure Environment Variables
Update your `.env` file with:

```bash
# LangSmith Configuration
LANGSMITH_API_KEY=your_actual_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=AroBot
```

### 3. Restart the Application
```bash
python app.py
```

You should see: `✅ LangSmith tracing initialized successfully` in the logs.

## 📊 What Gets Traced

### Core Workflow Functions
- **`process_request`** - Main entry point for all requests
- **`select_tools`** - Tool selection logic and LLM decision making
- **`execute_tools`** - Tool execution with timing and results
- **`generate_final_response`** - Response synthesis and formatting

### Individual Tools
- **`analyze_text_tool`** - Medical text analysis with RAG
- **`analyze_image_tool`** - Image processing and OCR
- **`get_medicine_info_tool`** - Medicine information retrieval
- **`web_search_tool`** - Web search operations
- **`get_weather_tool`** - Weather API calls
- **`access_memory_tool`** - Conversation memory access

### LLM Operations
- **`generate_text_response`** - Text generation calls
- **`generate_vision_response`** - Vision model calls
- **`answer_medical_query`** - Medical query processing
- **`answer_medicine`** - Medicine-specific responses
- **`smart_response`** - Smart routing logic

### Memory & Context
- **`initialize_session`** - New session creation
- **`add_user_message`** - User input logging
- **`add_assistant_response`** - AI response logging
- **`get_conversation_context`** - Context retrieval
- **`record_prescription_analysis`** - Prescription data storage
- **`record_medical_query`** - Medical query tracking

### Web Search & External APIs
- **`web_search`** - General web searches
- **`search_medicine_info`** - Medicine-specific searches
- **`search_disease_info`** - Disease information searches
- **`search_medical_news`** - Medical news searches

## 🎯 LangSmith Dashboard Views

### 1. Traces Tab - Real-time Monitoring
View complete request flows including:
- Session initialization
- User message processing
- Tool selection reasoning
- Individual tool executions
- LLM calls with prompts/responses
- Final response generation
- Memory updates

### 2. Analytics Tab - Performance Metrics
Monitor:
- Request volume and patterns
- Average response times
- Success/failure rates
- Tool usage statistics
- Peak usage times
- Common query types

### 3. Debugging Tab - Error Analysis
Track:
- Failed operations with stack traces
- Performance bottlenecks
- Tool execution failures
- LLM response issues
- Pattern analysis for improvements

### 4. Cost Tracking - Token Usage
Monitor:
- Token consumption per request
- Cost per session
- Optimization opportunities
- Usage trends over time

## 🔧 Advanced Configuration

### Custom Project Names
Set different project names for different environments:

```bash
# Development
LANGCHAIN_PROJECT=AroBot-Dev

# Staging  
LANGCHAIN_PROJECT=AroBot-Staging

# Production
LANGCHAIN_PROJECT=AroBot-Prod
```

### Selective Tracing
To disable tracing for specific environments:

```bash
LANGCHAIN_TRACING_V2=false
```

### Custom Metadata
The tracing includes rich metadata:
- Session IDs
- Tool execution times
- Input/output data
- Error information
- Context data

## 📈 Monitoring Best Practices

### 1. Regular Health Checks
- Monitor the Analytics tab for unusual patterns
- Check error rates in the Debugging tab
- Review cost trends in Cost Tracking

### 2. Performance Optimization
- Identify slow tools from traces
- Optimize frequently used code paths
- Monitor token usage for cost control

### 3. Error Investigation
- Use trace comparison for debugging
- Analyze error patterns
- Set up alerts for critical failures

## 🛠️ Troubleshooting

### Issue: "LangSmith tracing not available"
**Solutions:**
1. Check your `LANGSMITH_API_KEY` is set correctly
2. Verify `LANGCHAIN_TRACING_V2=true` in your `.env`
3. Ensure network connectivity to `https://api.smith.langchain.com`

### Issue: Traces not appearing in dashboard
**Solutions:**
1. Wait 1-2 minutes for traces to appear
2. Check the correct project is selected
3. Verify API key permissions
4. Check application logs for tracing errors

### Issue: High token costs
**Solutions:**
1. Review token usage in Cost Tracking tab
2. Optimize prompt lengths
3. Implement response caching
4. Reduce RAG context size for simple queries

## 🎯 Example Trace Flow

For a typical request "What are the side effects of paracetamol?":

```
🔄 process_request (2.3s total)
├─ 📝 initialize_session (45ms)
├─ 🔍 select_tools (120ms)
│  └─ Selected: get_medicine_info_tool
├─ ⚙️ execute_tools (1.8s)
│  └─ 💊 get_medicine_info_tool (1.8s)
│     ├─ RAG context retrieval (400ms)
│     ├─ Web search (600ms)
│     └─ LLM response generation (800ms)
├─ 📝 generate_final_response (200ms)
├─ 💾 add_user_message (15ms)
├─ 💾 add_assistant_response (20ms)
└─ 📊 record_medical_query (10ms)
```

## 📚 Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Tracing Guide](https://python.langchain.com/docs/langsmith/walkthrough)
- [AroBot Architecture Overview](./WORKFLOW_DIAGRAM.txt)

## 🎉 Success Indicators

When LangSmith is working correctly, you'll see:

1. **Application logs**: `✅ LangSmith tracing initialized successfully`
2. **LangSmith dashboard**: New traces appearing for each request
3. **Complete workflow visibility**: All steps from request to response
4. **Rich metadata**: Session IDs, timing, input/output data
5. **Error tracking**: Detailed error information when issues occur

Your AroBot system now has comprehensive observability! 🚀 