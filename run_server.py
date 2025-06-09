#!/usr/bin/env python3
import asyncio
import sys
from mcp.server.stdio import stdio_server

# Simple import since everything is in root
from server import OracleLogsMCPServer

async def main():
    """Run the Oracle Logs MCP Server"""
    print("🚀 Starting Oracle Logs MCP Server...")
    print("📋 Available tools:")
    print("  - search_logs_by_country")
    print("  - search_logs_by_location") 
    print("  - search_logs_by_ip")
    print("  - get_traffic_analytics")
    print("💡 Server ready for MCP connections...")
    
    # Create the server instance
    oracle_server = OracleLogsMCPServer()
    
    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await oracle_server.server.run(
            read_stream,
            write_stream,
            {}  # Empty initialization options
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
