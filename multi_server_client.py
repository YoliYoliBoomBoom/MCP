#!/usr/bin/env python3
"""
Multi-Server MCP Client with LlamaIndex and Ollama

This script demonstrates connecting to multiple MCP servers (database + weather) 
using a local LLM via Ollama. The setup includes:

- Database Server on http://127.0.0.1:8000/sse (people database)
- Weather Server on http://127.0.0.1:8001/sse (US weather data) 
- Local LLM via Ollama (llama3.2)
"""

import asyncio
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context

# Apply nest_asyncio for Jupyter compatibility (if needed)
nest_asyncio.apply()

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

async def setup_llm():
    """Setup the local LLM using Ollama"""
    print("🤖 Setting up Ollama LLM...")
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    Settings.llm = llm
    print("✅ LLM configured successfully!")
    return llm

async def connect_to_servers():
    """Connect to both MCP servers and get their tools"""
    print("🔌 Connecting to MCP servers...")
    
    # Connect to database server (port 8000)
    db_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    db_tools = McpToolSpec(client=db_client)
    
    # Connect to weather server (port 8001)
    weather_client = BasicMCPClient("http://127.0.0.1:8001/sse")
    weather_tools = McpToolSpec(client=weather_client)
    
    print("✅ Connected to both servers!")
    
    # Get tools from both servers
    print("\n📋 Getting available tools...")
    db_tool_list = await db_tools.to_tool_list_async()
    print(f"=== DATABASE TOOLS ({len(db_tool_list)}) ===")
    for tool in db_tool_list:
        print(f"  • {tool.metadata.name}")
    
    weather_tool_list = await weather_tools.to_tool_list_async()
    print(f"\n=== WEATHER TOOLS ({len(weather_tool_list)}) ===")
    for tool in weather_tool_list:
        print(f"  • {tool.metadata.name}")
    
    # Combine all tools
    all_tools = db_tool_list + weather_tool_list
    print(f"\n✅ Total tools available: {len(all_tools)}")
    
    return all_tools

async def create_agent(all_tools, llm):
    """Create the multi-server agent"""
    print("\n🛠️ Creating multi-server agent...")
    agent = FunctionAgent(
        name="Multi-Server Agent",
        description="An agent that can work with both weather data and database operations.",
        tools=all_tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    
    agent_context = Context(agent)
    print(f"✅ Multi-server agent created with {len(all_tools)} tools!")
    return agent, agent_context

async def handle_user_message(message_content, agent, agent_context, verbose=True):
    """Handle user message and return agent response"""
    handler = agent.run(message_content, ctx=agent_context)
    
    async for event in handler.stream_events():
        if verbose and type(event) == ToolCall:
            print(f"🔧 Calling tool: {event.tool_name}")
            print(f"   Parameters: {event.tool_kwargs}")
        elif verbose and type(event) == ToolCallResult:
            print(f"✅ Tool result: {str(event.tool_output)[:200]}...")
    
    response = await handler
    return str(response)

async def interactive_chat(agent, agent_context):
    """Run interactive chat with the agent"""
    print("\n" + "="*60)
    print("🚀 Multi-Server MCP Client Ready!")
    print("="*60)
    print("You can now ask questions about:")
    print("  📊 Database: Add/query people data")
    print("  🌦️  Weather: Get alerts/forecasts for US locations")
    print("  🔄 Combined: Mix database and weather operations")
    print("\nType 'exit' to quit, 'help' for examples")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 Your message: ").strip()
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("\n📝 Example commands:")
                print("  Database:")
                print("    • Add John Doe, age 30, engineer to the database")
                print("    • Show me all people in the database")
                print("    • Find all people over 25 years old")
                print("  Weather:")
                print("    • Get weather alerts for California")
                print("    • Get forecast for NYC (40.7128, -74.0060)")
                print("    • Check alerts for Texas")
                print("  Combined:")
                print("    • Add a meteorologist to the database, then check weather in NY")
                continue
                
            if not user_input:
                print("Please enter a message or 'exit' to quit.")
                continue
            
            print(f"\n🤔 Processing: {user_input}")
            print("-" * 50)
            
            response = await handle_user_message(user_input, agent, agent_context, verbose=True)
            
            print("-" * 50)
            print(f"🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'exit' to quit.")

async def main():
    """Main function to run the multi-server client"""
    try:
        # Setup LLM
        llm = await setup_llm()
        
        # Connect to servers and get tools
        all_tools = await connect_to_servers()
        
        # Create agent
        agent, agent_context = await create_agent(all_tools, llm)
        
        # Start interactive chat
        await interactive_chat(agent, agent_context)
        
    except Exception as e:
        print(f"❌ Error starting client: {e}")
        print("\nMake sure both MCP servers are running:")
        print("  • Database server: uv run python db_server.py --server_type=sse")
        print("  • Weather server: uv run python weather_server.py --server_type=sse")

if __name__ == "__main__":
    print("🌟 Starting Multi-Server MCP Client...")
    asyncio.run(main()) 