#!/usr/bin/env python3
"""
Multi-Server MCP Client - Streamlit Web Interface

A simple web UI for connecting to multiple MCP servers (database + weather) 
using a local LLM via Ollama.
"""

import asyncio
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context

# System prompt for the multi-server agent
SYSTEM_PROMPT = """\
You are an AI assistant with access to both weather information and database operations.

Available capabilities:
- Database operations: Add and read people data (name, age, profession)
- Weather information: Get weather alerts and forecasts for US locations

Tools available:
- add_data(query): Add people to database using SQL INSERT
- read_data(query): Query people database using SQL SELECT  
- get_alerts(state): Get weather alerts for US states (use 2-letter codes like "CA", "NY")
- get_forecast(latitude, longitude): Get weather forecast for coordinates

You can help with database management, weather queries, or both!
"""

@st.cache_resource
def setup_llm():
    """Setup the local LLM using Ollama - cached to avoid re-initialization"""
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    Settings.llm = llm
    return llm

def run_async(coro):
    """Helper function to run async code in Streamlit's event loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If loop is already running, we need to run in a thread
        import concurrent.futures
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    else:
        return loop.run_until_complete(coro)

async def async_connect_to_servers():
    """Connect to both MCP servers and get their tools"""
    # Connect to database server (port 8000)
    db_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    db_tools = McpToolSpec(client=db_client)
    
    # Connect to weather server (port 8001)
    weather_client = BasicMCPClient("http://127.0.0.1:8001/sse")
    weather_tools = McpToolSpec(client=weather_client)
    
    # Get tools from both servers
    db_tool_list = await db_tools.to_tool_list_async()
    weather_tool_list = await weather_tools.to_tool_list_async()
    all_tools = db_tool_list + weather_tool_list
    
    return all_tools

def connect_to_servers():
    """Sync wrapper for async_connect_to_servers"""
    return run_async(async_connect_to_servers())

async def async_create_agent(all_tools, llm):
    """Create the multi-server agent"""
    agent = FunctionAgent(
        name="Multi-Server Agent",
        description="An agent that can work with both weather data and database operations.",
        tools=all_tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, Context(agent)

def create_agent(all_tools, llm):
    """Sync wrapper for async_create_agent"""
    return run_async(async_create_agent(all_tools, llm))

async def async_handle_user_message(message_content, agent, agent_context):
    """Handle user message and return agent response with tool calls"""
    tool_calls = []
    handler = agent.run(message_content, ctx=agent_context)
    
    async for event in handler.stream_events():
        if type(event) == ToolCall:
            tool_calls.append(f"🔧 Calling: {event.tool_name}")
        elif type(event) == ToolCallResult:
            tool_calls.append(f"✅ Result: {str(event.tool_output)[:100]}...")
    
    response = await handler
    return str(response), tool_calls

def handle_user_message(message_content, agent, agent_context):
    """Sync wrapper for async_handle_user_message"""
    return run_async(async_handle_user_message(message_content, agent, agent_context))

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Multi-Server MCP Client",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Server MCP Client")
    st.markdown("**Connect to Database + Weather servers using Ollama LLM**")
    
    # Sidebar with server status and info
    with st.sidebar:
        st.header("🔌 Server Status")
        
        # Server status indicators
        st.write("**Database Server:** `127.0.0.1:8000`")
        st.write("**Weather Server:** `127.0.0.1:8001`")
        st.write("**LLM:** Ollama (llama3.2)")
        
        st.markdown("---")
        st.header("💡 Example Commands")
        
        st.markdown("**Database:**")
        st.code("Add John Doe, age 30, engineer to database")
        st.code("Show all people in database")
        
        st.markdown("**Weather:**")
        st.code("Get weather alerts for California")
        st.code("Get forecast for NYC (40.7128, -74.0060)")
        
        st.markdown("**Combined:**")
        st.code("Add Tom the 60 yo meteorologist to DB, then check NY weather")
        
        # Refresh button
        if st.button("🔄 Refresh Connection"):
            st.session_state.clear()
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    # Initialize agent if not done
    if not st.session_state.agent_initialized:
        with st.spinner("🚀 Initializing MCP client..."):
            try:
                # Setup LLM
                llm = setup_llm()
                st.success("✅ LLM configured!")
                
                # Connect to servers and get tools
                all_tools = connect_to_servers()
                st.success("✅ Connected to servers!")
                
                # Create agent
                agent, agent_context = create_agent(all_tools, llm)
                
                # Store in session state
                st.session_state.agent = agent
                st.session_state.agent_context = agent_context
                st.session_state.all_tools = all_tools
                st.session_state.agent_initialized = True
                
                st.success(f"✅ Agent ready with {len(all_tools)} tools!")
                
            except Exception as e:
                st.error(f"❌ Error initializing: {e}")
                st.error("Make sure both MCP servers are running!")
                with st.expander("💡 How to start servers"):
                    st.code("uv run python db_server.py --server_type=sse")
                    st.code("uv run python weather_server.py --server_type=sse")
                st.stop()
    
    # Display chat messages
    st.header("💬 Chat")
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tool_calls" in message and message["tool_calls"]:
                with st.expander("🔧 Tool Calls"):
                    for tool_call in message["tool_calls"]:
                        st.text(tool_call)
    
    # Chat input
    if prompt := st.chat_input("Ask about database or weather..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing..."):
                try:
                    response, tool_calls = handle_user_message(
                        prompt, 
                        st.session_state.agent, 
                        st.session_state.agent_context
                    )
                    
                    st.markdown(response)
                    
                    # Show tool calls if any
                    if tool_calls:
                        with st.expander("🔧 Tool Calls"):
                            for tool_call in tool_calls:
                                st.text(tool_call)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "tool_calls": tool_calls
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 