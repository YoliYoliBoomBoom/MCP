#!/usr/bin/env python3
"""Test script for the weather MCP server."""

import asyncio
import json
import subprocess
import sys

async def test_weather_server():
    """Test the weather server by sending MCP messages."""
    
    # Start the weather server
    process = subprocess.Popen(
        [sys.executable, "weather_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Initialize the server
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        print("Sending initialize message...")
        process.stdin.write(json.dumps(init_message) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        if response:
            print("Server response:", response.strip())
        
        # Test the forecast tool
        forecast_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_forecast",
                "arguments": {
                    "latitude": 40.7128,
                    "longitude": -74.0060
                }
            }
        }
        
        print("\nTesting forecast for NYC...")
        process.stdin.write(json.dumps(forecast_message) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        if response:
            print("Forecast response:", response.strip())
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        process.wait()

def main():
    print("Testing Weather MCP Server")
    print("=" * 30)
    asyncio.run(test_weather_server())

if __name__ == "__main__":
    main() 