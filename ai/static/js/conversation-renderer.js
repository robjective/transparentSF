/**
 * Standardized Conversation Renderer Component
 * 
 * Renders agent conversations consistently across the application
 * Based on the backend.html implementation with enhancements for reusability
 */

class ConversationRenderer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }
        
        this.options = {
            showTimestamps: options.showTimestamps || false,
            enableMarkdown: options.enableMarkdown !== false, // default true
            enableToolCallDetails: options.enableToolCallDetails !== false, // default true
            enableChartProcessing: options.enableChartProcessing !== false, // default true
            autoScroll: options.autoScroll !== false, // default true
            className: options.className || 'conversation-renderer',
            ...options
        };
        
        this.toolCallData = new Map();
        this.init();
    }
    
    init() {
        // Add the conversation renderer class
        this.container.classList.add(this.options.className);
        
        // Ensure marked.js is available if markdown is enabled
        if (this.options.enableMarkdown && typeof marked === 'undefined') {
            console.warn('marked.js not available - markdown rendering disabled');
            this.options.enableMarkdown = false;
        }
    }
    
    /**
     * Render a complete conversation from session data
     */
    renderConversation(sessionData) {
        this.clear();
        
        if (!sessionData) {
            this.addSystemMessage("No conversation data available");
            return;
        }
        
        // Add session info if available
        if (sessionData.session_id && this.options.showTimestamps) {
            const startTime = sessionData.start_time || sessionData.timestamp;
            const model = sessionData.model || 'Unknown';
            const endTime = sessionData.end_time || '';
            let timeInfo = new Date(startTime).toLocaleString();
            if (endTime) {
                timeInfo += ` - ${new Date(endTime).toLocaleString()}`;
            }
            this.addSystemMessage(`Session: ${sessionData.session_id} | Model: ${model} | ${timeInfo}`);
        }
        
        // Process messages and tool calls in chronological order
        const events = this.buildEventTimeline(sessionData);
        
        // Process events in chronological order
        // console.log('Processing events:', events.length, 'events');
        events.forEach((event, index) => {
            // console.log(`Event ${index}:`, event.type, event);
            switch (event.type) {
                case 'message':
                    this.addMessage(event.content, event.sender, this.options.enableMarkdown, event.timestamp);
                    break;
                case 'tool_call_start':
                    // For session replay, we'll wait for the end event to show the completed state
                    break;
                case 'tool_call_end':
                    // Show completed tool call with details
                    this.addCompletedToolCall(event.tool_name, event.success, event.execution_time_ms, event.response, event.arguments);
                    break;
            }
        });
    }
    
    /**
     * Build a chronological timeline of conversation events
     */
    buildEventTimeline(sessionData) {
        const events = [];
        
        // Choose between conversation array approach or intermediate_responses approach
        // If we have intermediate_responses, use those for a cleaner streaming replay
        // This avoids duplication issues with final_response and conversation array
        const useIntermediateResponses = sessionData.intermediate_responses && 
            Array.isArray(sessionData.intermediate_responses) && 
            sessionData.intermediate_responses.length > 0;
        
        // Debug: Session data analysis (uncomment for debugging)
        // console.log('Session data analysis:', {
        //     hasIntermediateResponses: !!sessionData.intermediate_responses,
        //     intermediateResponsesCount: sessionData.intermediate_responses?.length || 0,
        //     hasConversation: !!sessionData.conversation,
        //     conversationCount: sessionData.conversation?.length || 0,
        //     hasToolCalls: !!sessionData.tool_calls,
        //     toolCallsCount: sessionData.tool_calls?.length || 0,
        //     useIntermediateResponses: useIntermediateResponses
        // });
        
        if (useIntermediateResponses) {
            // Use intermediate responses approach - properly interleaved with tool calls
            
            // First add user message if present
            if (sessionData.user_input) {
                events.push({
                    type: 'message',
                    content: sessionData.user_input,
                    sender: 'user',
                    timestamp: sessionData.start_time || sessionData.timestamp
                });
            }
            
            // Add each intermediate response as a separate message (to preserve interleaving)
            sessionData.intermediate_responses.forEach((response, index) => {
                events.push({
                    type: 'message',
                    content: response.content,
                    sender: 'assistant',
                    timestamp: response.timestamp
                });
            });
            
            // Add tool calls - will be properly sorted with responses by timestamp
            if (sessionData.tool_calls && Array.isArray(sessionData.tool_calls)) {
                sessionData.tool_calls.forEach((toolCall, index) => {
                    // Use start_time if available (Unix timestamp), convert to ISO
                    const timestamp = toolCall.start_time ? 
                        new Date(toolCall.start_time * 1000).toISOString() : 
                        sessionData.start_time || new Date().toISOString();
                    
                    events.push({
                        type: 'tool_call_end',  // Show completed tool calls
                        tool_name: toolCall.tool_name,
                        tool_id: `tool_${toolCall.tool_name}_${index}`,
                        success: toolCall.success,
                        response: toolCall.result,
                        arguments: toolCall.arguments,
                        execution_time_ms: toolCall.execution_time_ms,
                        timestamp: timestamp
                    });
                });
            }
        } else {
            // Fallback to conversation array approach (for sessions without intermediate_responses)
            if (sessionData.conversation && Array.isArray(sessionData.conversation)) {
                let toolCallIndex = 0;
                
                sessionData.conversation.forEach((msg, index) => {
                    if (msg.role === 'tool_call') {
                        // Handle tool call messages
                        const toolName = msg.content.replace('Tool: ', '');
                        let toolCallDetails = sessionData.tool_calls && toolCallIndex < sessionData.tool_calls.length ? 
                            sessionData.tool_calls[toolCallIndex] : null;
                        
                        if (toolCallDetails && toolCallDetails.tool_name === toolName) {
                            toolCallIndex++;
                        } else {
                            toolCallDetails = sessionData.tool_calls ? 
                                sessionData.tool_calls.find(tc => tc.tool_name === toolName) : null;
                        }
                        
                        if (toolCallDetails) {
                            events.push({
                                type: 'tool_call_end',
                                tool_name: toolCallDetails.tool_name,
                                tool_id: `tool_${toolCallDetails.tool_name}_${index}`,
                                success: toolCallDetails.success,
                                response: toolCallDetails.result,
                                arguments: toolCallDetails.arguments,
                                execution_time_ms: toolCallDetails.execution_time_ms,
                                timestamp: msg.timestamp
                            });
                        }
                    } else {
                        // Handle regular messages (user, assistant)
                        events.push({
                            type: 'message',
                            content: msg.content,
                            sender: msg.role,
                            timestamp: msg.timestamp
                        });
                    }
                });
            }
            
            // If no conversation array either, try final_response as last resort
            if ((!sessionData.conversation || sessionData.conversation.length === 0) && 
                sessionData.final_response && typeof sessionData.final_response === 'string' && 
                sessionData.final_response.trim()) {
                events.push({
                    type: 'message',
                    content: sessionData.final_response.trim(),
                    sender: 'assistant',
                    timestamp: sessionData.end_time || sessionData.timestamp || new Date().toISOString()
                });
            }
        }
        
        // Sort events by timestamp if available
        events.sort((a, b) => {
            if (a.timestamp && b.timestamp) {
                const timeA = typeof a.timestamp === 'number' ? a.timestamp * 1000 : new Date(a.timestamp).getTime();
                const timeB = typeof b.timestamp === 'number' ? b.timestamp * 1000 : new Date(b.timestamp).getTime();
                return timeA - timeB;
            }
            return 0;
        });
        
        return events;
    }
    
    /**
     * Add a message to the conversation
     */
    addMessage(content, sender = 'assistant', isMarkdown = true, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `conversation-message ${sender}`;
        
        if (this.options.showTimestamps) {
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'conversation-timestamp';
            if (timestamp) {
                timestampDiv.textContent = new Date(timestamp).toLocaleTimeString();
            } else {
                timestampDiv.textContent = new Date().toLocaleTimeString();
            }
            messageDiv.appendChild(timestampDiv);
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'conversation-content';
        
        // Store the raw content for streaming updates
        messageDiv._rawContent = content;
        
        // Process content
        let processedContent = content;
        if (sender === 'assistant' && this.options.enableChartProcessing) {
            processedContent = this.processChartPlaceholders(content);
        }
        
        if (isMarkdown && this.options.enableMarkdown) {
            contentDiv.innerHTML = marked.parse(processedContent);
        } else {
            // Convert newlines to <br> tags for non-markdown content
            processedContent = processedContent.replace(/\n/g, '<br>');
            contentDiv.innerHTML = processedContent;
        }
        
        messageDiv.appendChild(contentDiv);
        this.container.appendChild(messageDiv);
        
        if (this.options.autoScroll) {
            this.scrollToBottom();
        }
        
        return messageDiv;
    }
    
    /**
     * Append content to an existing message (for streaming)
     */
    appendToMessage(messageDiv, additionalContent) {
        if (!messageDiv || messageDiv._rawContent === undefined) {
            console.warn('Cannot append to message: invalid message element');
            return;
        }
        
        // Accumulate the raw content
        messageDiv._rawContent += additionalContent;
        
        // Re-render the entire content with proper markdown
        const contentDiv = messageDiv.querySelector('.conversation-content');
        if (contentDiv) {
            let processedContent = messageDiv._rawContent;
            
            if (this.options.enableChartProcessing) {
                processedContent = this.processChartPlaceholders(processedContent);
            }
            
            if (this.options.enableMarkdown && typeof marked !== 'undefined') {
                contentDiv.innerHTML = marked.parse(processedContent);
            } else {
                contentDiv.innerHTML = processedContent.replace(/\n/g, '<br>');
            }
            
            if (this.options.autoScroll) {
                this.scrollToBottom();
            }
        }
    }
    
    /**
     * Add a completed tool call with details (for session replay)
     */
    addCompletedToolCall(toolName, success = true, executionTimeMs = null, response = null, args = null) {
        const toolId = `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const toolCallDiv = document.createElement('div');
        toolCallDiv.id = toolId;
        toolCallDiv.className = `conversation-tool-call ${success ? 'completed' : 'error'}`;
        
        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'conversation-tool-call-content';
        
        const toolNameDiv = document.createElement('div');
        toolNameDiv.className = 'tool-name';
        toolNameDiv.textContent = `🔧 ${toolName}`;
        
        const toolStatusDiv = document.createElement('div');
        toolStatusDiv.className = 'tool-status';
        let statusText = success ? 'Success' : 'Failed';
        if (executionTimeMs) {
            statusText += ` (${executionTimeMs}ms)`;
        }
        toolStatusDiv.textContent = statusText;
        
        // Add queryURL link if available
        if (response && response.queryURL) {
            const queryUrlDiv = document.createElement('div');
            queryUrlDiv.className = 'tool-query-url';
            const link = document.createElement('a');
            link.href = response.queryURL;
            link.target = '_blank';
            link.textContent = '🔗 View Query';
            link.title = 'Click to view the actual API query in a new tab';
            queryUrlDiv.appendChild(link);
            contentDiv.appendChild(queryUrlDiv);
        }
        
        contentDiv.appendChild(toolNameDiv);
        contentDiv.appendChild(toolStatusDiv);
        toolCallDiv.appendChild(contentDiv);
        
        // Create details container if enabled
        if (this.options.enableToolCallDetails) {
            const detailsDiv = document.createElement('div');
            detailsDiv.className = 'conversation-tool-call-details';
            detailsDiv.innerHTML = `
                <h4>Tool Call Details</h4>
                <div><strong>Function:</strong> ${toolName}</div>
                <div><strong>Status:</strong> ${statusText}</div>
                <div><strong>Arguments:</strong> <pre>${args ? JSON.stringify(args, null, 2) : 'N/A'}</pre></div>
                <div><strong>Response:</strong> <pre>${response ? JSON.stringify(response, null, 2) : (success ? 'Success' : 'Failed')}</pre></div>
            `;
            toolCallDiv.appendChild(detailsDiv);
            
            // Add click handler to toggle details
            contentDiv.addEventListener('click', function() {
                detailsDiv.classList.toggle('show');
            });
        }
        
        this.container.appendChild(toolCallDiv);
        
        if (this.options.autoScroll) {
            this.scrollToBottom();
        }
        
        return toolCallDiv;
    }

    /**
     * Add a tool call to the conversation
     */
    addToolCall(toolName, toolId = null, response = null) {
        const id = toolId || `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        const toolCallDiv = document.createElement('div');
        toolCallDiv.id = id;
        toolCallDiv.className = 'conversation-tool-call running';
        
        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'conversation-tool-call-content';
        
        const toolNameDiv = document.createElement('div');
        toolNameDiv.className = 'tool-name';
        toolNameDiv.textContent = `🔧 ${toolName}`;
        
        const toolStatusDiv = document.createElement('div');
        toolStatusDiv.className = 'tool-status';
        toolStatusDiv.textContent = 'Running';
        
        // Add queryURL link if available
        if (response && response.queryURL) {
            const queryUrlDiv = document.createElement('div');
            queryUrlDiv.className = 'tool-query-url';
            const link = document.createElement('a');
            link.href = response.queryURL;
            link.target = '_blank';
            link.textContent = '🔗 View Query';
            link.title = 'Click to view the actual API query in a new tab';
            queryUrlDiv.appendChild(link);
            contentDiv.appendChild(queryUrlDiv);
        }
        
        contentDiv.appendChild(toolNameDiv);
        contentDiv.appendChild(toolStatusDiv);
        toolCallDiv.appendChild(contentDiv);
        
        // Create details container if enabled
        if (this.options.enableToolCallDetails) {
            const detailsDiv = document.createElement('div');
            detailsDiv.className = 'conversation-tool-call-details';
            detailsDiv.innerHTML = `
                <h4>Tool Call Details</h4>
                <div><strong>Function:</strong> ${toolName}</div>
                <div><strong>Status:</strong> Running...</div>
                <div><strong>Arguments:</strong> <pre id="${id}-args">Loading...</pre></div>
                <div><strong>Response:</strong> <pre id="${id}-response">Waiting for completion...</pre></div>
            `;
            toolCallDiv.appendChild(detailsDiv);
            
            // Add click handler to toggle details
            contentDiv.addEventListener('click', function() {
                detailsDiv.classList.toggle('show');
            });
        }
        
        this.container.appendChild(toolCallDiv);
        
        // Store tool call data
        this.toolCallData.set(id, {
            name: toolName,
            status: 'running',
            arguments: null,
            response: null,
            startTime: Date.now()
        });
        
        if (this.options.autoScroll) {
            this.scrollToBottom();
        }
        
        return id;
    }
    
    /**
     * Complete a tool call
     */
    completeToolCall(toolId, success = true, response = null, executionTimeMs = null) {
        const toolCallDiv = document.getElementById(toolId);
        if (!toolCallDiv) return;
        
        // Update visual state
        toolCallDiv.className = `conversation-tool-call ${success ? 'completed' : 'error'}`;
        
        const statusDiv = toolCallDiv.querySelector('.tool-status');
        if (statusDiv) {
            let statusText = success ? 'Success' : 'Failure';
            if (executionTimeMs) {
                statusText += ` (${executionTimeMs}ms)`;
            }
            statusDiv.textContent = statusText;
        }
        
        // Add queryURL link if available and not already added
        if (response && response.queryURL && !toolCallDiv.querySelector('.tool-query-url')) {
            const contentDiv = toolCallDiv.querySelector('.conversation-tool-call-content');
            if (contentDiv) {
                const queryUrlDiv = document.createElement('div');
                queryUrlDiv.className = 'tool-query-url';
                const link = document.createElement('a');
                link.href = response.queryURL;
                link.target = '_blank';
                link.textContent = '🔗 View Query';
                link.title = 'Click to view the actual API query in a new tab';
                queryUrlDiv.appendChild(link);
                contentDiv.appendChild(queryUrlDiv);
            }
        }
        
        // Update details if enabled
        if (this.options.enableToolCallDetails) {
            const detailsDiv = toolCallDiv.querySelector('.conversation-tool-call-details');
            if (detailsDiv) {
                const statusText = detailsDiv.querySelector('div:nth-child(2)');
                if (statusText) {
                    let status = success ? 'Success' : 'Failure';
                    if (executionTimeMs) {
                        status += ` (${executionTimeMs}ms)`;
                    }
                    statusText.innerHTML = `<strong>Status:</strong> ${status}`;
                }
                
                const responseElement = detailsDiv.querySelector(`#${toolId}-response`);
                if (responseElement) {
                    responseElement.textContent = response ? JSON.stringify(response, null, 2) : (success ? 'Success' : 'Failure');
                }
            }
        }
        
        // Update stored data
        if (this.toolCallData.has(toolId)) {
            const data = this.toolCallData.get(toolId);
            data.status = success ? 'success' : 'failure';
            data.response = response;
            data.endTime = Date.now();
            data.duration = data.endTime - data.startTime;
        }
    }
    
    /**
     * Update tool call arguments
     */
    updateToolCallArguments(toolId, args) {
        if (!this.options.enableToolCallDetails) return;
        
        const argsElement = document.getElementById(`${toolId}-args`);
        if (argsElement) {
            argsElement.textContent = JSON.stringify(args, null, 2);
        }
        
        // Update stored data
        if (this.toolCallData.has(toolId)) {
            this.toolCallData.get(toolId).arguments = args;
        }
    }
    
    /**
     * Add a system message
     */
    addSystemMessage(content) {
        return this.addMessage(content, 'system', false);
    }
    
    /**
     * Clear the conversation
     */
    clear() {
        this.container.innerHTML = '';
        this.toolCallData.clear();
    }
    
    /**
     * Scroll to bottom of conversation
     */
    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }
    
    /**
     * Process chart placeholders (from backend.html)
     */
    processChartPlaceholders(content) {
        if (!this.options.enableChartProcessing) return content;
        
        return content.replace(/\[CHART:(\w+):(\d+)\]/g, (match, type, id) => {
            // Generate correct URLs based on chart type
            let chartUrl;
            if (type === 'time_series_id') {
                chartUrl = `/backend/time-series-chart?chart_id=${id}`;
            } else if (type === 'map') {
                chartUrl = `/backend/map-chart?id=${id}`;
            } else if (type === 'anomaly') {
                chartUrl = `/anomaly-analyzer/anomaly-chart?id=${id}`;
            } else if (type === 'time_series') {
                // Handle time_series with parameters (metric_id:district_id:period_type)
                const params = id.split(':');
                if (params.length === 3) {
                    const [metric_id, district_id, period_type] = params;
                    chartUrl = `/backend/time-series-chart?metric_id=${metric_id}&district_id=${district_id}&period_type=${period_type}`;
                } else {
                    // Fallback for malformed time_series parameters
                    chartUrl = `/backend/time-series-chart?chart_id=${id}`;
                }
            } else {
                // Fallback to generic chart endpoint
                chartUrl = `/backend/charts/${type}/${id}`;
            }
            
            return `<div class="chart-container">
                <iframe src="${chartUrl}" width="100%" height="400" frameborder="0"></iframe>
                <div class="chart-caption">Chart ${id} (${type})</div>
            </div>`;
        });
    }
    
    /**
     * Export conversation data
     */
    exportData() {
        return {
            toolCalls: Object.fromEntries(this.toolCallData),
            html: this.container.innerHTML,
            timestamp: new Date().toISOString()
        };
    }
    
    /**
     * Get conversation summary
     */
    getSummary() {
        const messages = this.container.querySelectorAll('.conversation-message');
        const toolCalls = this.container.querySelectorAll('.conversation-tool-call');
        
        return {
            messageCount: messages.length,
            toolCallCount: toolCalls.length,
            successfulToolCalls: this.container.querySelectorAll('.conversation-tool-call.completed').length,
            failedToolCalls: this.container.querySelectorAll('.conversation-tool-call.error').length
        };
    }
}

// Global utility functions for backwards compatibility
window.ConversationRenderer = ConversationRenderer;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConversationRenderer;
}
