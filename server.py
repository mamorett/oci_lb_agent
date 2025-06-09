import asyncio
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

# Simple imports since everything is in root
from oracle_client import OracleLogsClient
from models import LogEntry, LogResponse

class OracleLogsMCPServer:
    def __init__(self):
        self.server = Server("oracle-logs-mcp")
        self.oracle_client = OracleLogsClient()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="search_logs_by_country",
                    description="Search Oracle Cloud logs by country or country code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {"type": "string", "description": "Country name (e.g., 'United States')"},
                            "country_code": {"type": "string", "description": "Country code (e.g., 'US')"},
                            "time_range": {"type": "string", "description": "Time range (e.g., '24h', '7d')", "default": "24h"},
                            "limit": {"type": "integer", "description": "Maximum number of results", "default": 10000}
                        }
                    }
                ),
                Tool(
                    name="search_logs_by_location",
                    description="Search Oracle Cloud logs within geographic bounds",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lat_min": {"type": "number", "description": "Minimum latitude"},
                            "lat_max": {"type": "number", "description": "Maximum latitude"},
                            "lon_min": {"type": "number", "description": "Minimum longitude"},
                            "lon_max": {"type": "number", "description": "Maximum longitude"},
                            "time_range": {"type": "string", "description": "Time range", "default": "24h"},
                            "limit": {"type": "integer", "description": "Maximum results", "default": 10000}
                        },
                        "required": ["lat_min", "lat_max", "lon_min", "lon_max"]
                    }
                ),
                Tool(
                    name="search_logs_by_ip",
                    description="Search Oracle Cloud logs by IP address or range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ip_address": {"type": "string", "description": "Specific IP address"},
                            "ip_range": {"type": "string", "description": "IP range in CIDR notation"},
                            "time_range": {"type": "string", "description": "Time range", "default": "24h"},
                            "limit": {"type": "integer", "description": "Maximum results", "default": 10000}
                        }
                    }
                ),
                Tool(
                    name="get_traffic_analytics",
                    description="Get aggregated traffic analytics from Oracle Cloud logs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "time_range": {"type": "string", "description": "Time range", "default": "24h"},
                            "group_by": {"type": "string", "description": "Group by field", "enum": ["country", "city", "isp", "protocol"], "default": "country"},
                            "limit": {"type": "integer", "description": "Maximum results for analysis", "default": 10000}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "search_logs_by_country":
                    logs = await self.oracle_client.search_logs_by_country(arguments)
                    response = LogResponse(logs=logs, total_count=len(logs))
                    return [TextContent(type="text", text=json.dumps(response.dict(), indent=2, default=str))]
                
                elif name == "search_logs_by_location":
                    logs = await self.oracle_client.search_logs_by_location(arguments)
                    response = LogResponse(logs=logs, total_count=len(logs))
                    return [TextContent(type="text", text=json.dumps(response.dict(), indent=2, default=str))]
                
                elif name == "search_logs_by_ip":
                    logs = await self.oracle_client.search_logs_by_ip(arguments)
                    response = LogResponse(logs=logs, total_count=len(logs))
                    return [TextContent(type="text", text=json.dumps(response.dict(), indent=2, default=str))]
                
                elif name == "get_traffic_analytics":
                    analytics = await self.oracle_client.get_traffic_analytics(arguments)
                    return [TextContent(type="text", text=json.dumps(analytics, indent=2, default=str))]
                
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                print(f"❌ {error_msg}")
                return [TextContent(type="text", text=error_msg)]
