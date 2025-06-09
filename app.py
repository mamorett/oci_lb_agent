import streamlit as st
from typing import Dict, Any, List
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import asyncio
import json
import re
from oracle_client import OracleLogsClient  # Your existing Oracle client
from langchain_ollama import ChatOllama  # Add this import at the top

@dataclass
class AgentState:
    query: str = ""
    context: str = ""
    analysis: str = ""
    response: str = ""
    next_action: str = ""
    iteration: int = 0
    max_iterations: int = 3
    oracle_data: dict = None  # New field for Oracle logs data
    intent: dict = None       # New field for user intent

class OracleGraphAIAgent:
    def __init__(self, use_ollama=True, ollama_model="llama3.1"):
        if use_ollama:
            # Use Ollama instead of Gemini
            self.llm = ChatOllama(
                model=ollama_model,
                temperature=0.7,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
            self.analyzer = ChatOllama(
                model=ollama_model,
                temperature=0.3,
                base_url="http://localhost:11434"
            )
        else:
            # Keep Gemini as fallback
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
            )
            self.analyzer = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
            )
        
        # Initialize Oracle client
        self.oracle_client = OracleLogsClient()

    def run_with_reasoning(self, chat_history: List[Dict[str, str]], st_container):
        # The latest user message is the query
        query = chat_history[-1]["content"]
        state = AgentState(query=query)

        # Prepare conversation history as LangChain messages
        lc_messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        # Router - Enhanced to detect Oracle logs queries
        st_container.markdown("🔀 **Router:** Analyzing query and checking for Oracle logs requests...")
        router_out = self._router_node(state, lc_messages)
        state = AgentState(**{**vars(state), **router_out})
        st_container.markdown(f"**Context:**\n{state.context}")

        # Intent Analysis - New step for Oracle logs
        if self._is_oracle_logs_query(state.query):
            st_container.markdown("🎯 **Intent Analyzer:** Parsing Oracle logs request...")
            intent_out = self._intent_analyzer_node(state)
            state = AgentState(**{**vars(state), **intent_out})
            st_container.markdown(f"**Intent:**\n{json.dumps(state.intent, indent=2)}")

            # Oracle Data Fetcher - New step
            st_container.markdown("📊 **Oracle Fetcher:** Retrieving logs data...")
            oracle_out = asyncio.run(self._oracle_fetcher_node(state))
            state = AgentState(**{**vars(state), **oracle_out})
            
            if state.oracle_data and state.oracle_data.get("type") != "error":
                if state.oracle_data["type"] == "logs":
                    st_container.markdown(f"**Found {state.oracle_data['count']} log entries**")
                elif state.oracle_data["type"] == "analytics":
                    analytics = state.oracle_data["data"]
                    st_container.markdown(f"**Analytics:** {analytics.get('total_requests', 0)} requests, {analytics.get('unique_ips', 0)} unique IPs")
            else:
                st_container.markdown("⚠️ **No data found or error occurred**")

        # Analyzer - Enhanced to handle Oracle data
        st_container.markdown("🧐 **Analyzer:** Determining response strategy...")
        analyzer_out = self._analyzer_node(state, lc_messages)
        state = AgentState(**{**vars(state), **analyzer_out})
        st_container.markdown(f"**Analysis:**\n{state.analysis}")

        # Decide next step
        if analyzer_out["next_action"] == "research" and not state.oracle_data:
            st_container.markdown("🔬 **Researcher:** Gathering additional information...")
            researcher_out = self._researcher_node(state, lc_messages)
            state = AgentState(**{**vars(state), **researcher_out})
            st_container.markdown(f"**Research:**\n{state.context.split('Research:',1)[-1].strip()}")

        # Responder - Enhanced to handle Oracle data
        st_container.markdown("💬 **Responder:** Generating answer...")
        full_response = self._generate_oracle_response(state, chat_history)
        state.response = full_response

        # Validator
        st_container.markdown("✅ **Validator:** Checking if the answer is complete...")
        validator_out = self._validator_node(state, lc_messages, full_response)
        state = AgentState(**{**vars(state), **validator_out})
        validation = state.context.split("Validation:",1)[-1].strip()
        st_container.markdown(f"**Validation:**\n{validation}")

        # Display final answer
        st_container.markdown("---")
        st_container.markdown("### 🎯 **Final Answer:**")
        response_placeholder = st_container.empty()
        
        # Stream the response
        displayed_response = ""
        for char in full_response:
            displayed_response += char
            response_placeholder.markdown(displayed_response + "▌")
        response_placeholder.markdown(full_response)

        return full_response

    def _is_oracle_logs_query(self, query: str) -> bool:
        """Check if the query is related to Oracle logs"""
        oracle_keywords = [
            "logs", "log", "oracle", "traffic", "ip", "country", "visitor",
            "request", "analytics", "unique", "location", "geographic"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in oracle_keywords)

    def _intent_analyzer_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyze user intent for Oracle logs queries"""
        intent = self._analyze_intent(state.query)
        return {"intent": intent}

    def _analyze_intent(self, message: str) -> dict:
        """Enhanced intent analysis with better time range extraction"""
        message_lower = message.lower()
        
        intent = {
            "action": "analytics",
            "params": {"time_range": "24h", "limit": 10000}
        }
        
        # Extract time ranges with regex
        time_range = self._extract_time_range(message_lower)
        if time_range:
            intent["params"]["time_range"] = time_range
        
        # Detect unique IP requests
        if any(phrase in message_lower for phrase in [
            "unique ip", "unique ips", "distinct ip", "different ip", 
            "how many ip", "ip addresses", "unique visitors", "distinct visitors"
        ]):
            intent["action"] = "analytics"
            intent["params"]["group_by"] = "ip"
        
        # Country search
        elif any(word in message_lower for word in ["country", "from"]):
            intent["action"] = "search_country"
            countries = ["united states", "usa", "germany", "france", "china", "russia", "sweden", "norway"]
            for country in countries:
                if country in message_lower:
                    intent["params"]["country"] = country.title()
        
        # IP search
        elif any(word in message_lower for word in ["ip", "address"]) and "unique" not in message_lower:
            intent["action"] = "search_ip"
            ip_match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message)
            if ip_match:
                intent["params"]["ip_address"] = ip_match.group()
        
        # Geographic search
        elif any(word in message_lower for word in ["location", "geographic", "lat", "lon"]):
            intent["action"] = "search_location"
        
        # Extract limit/count requests
        limit = self._extract_limit(message_lower)
        if limit:
            intent["params"]["limit"] = limit
        
        return intent

    def _extract_time_range(self, message: str) -> str:
        """Extract time range from natural language"""
        patterns = [
            (r'(?:last|past|previous)?\s*(\d+)\s*hours?', lambda m: f"{m.group(1)}h"),
            (r'(?:last|past|previous)?\s*(\d+)\s*days?', lambda m: f"{m.group(1)}d"),
            (r'(?:last|past|previous)?\s*(\d+)\s*weeks?', lambda m: f"{int(m.group(1)) * 7}d"),
            (r'(?:last|past|previous)?\s*(\d+)\s*months?', lambda m: f"{int(m.group(1)) * 30}d"),
            (r'yesterday', lambda m: "24h"),
            (r'last week', lambda m: "7d"),
            (r'past week', lambda m: "7d"),
            (r'this week', lambda m: "7d"),
            (r'last month', lambda m: "30d"),
            (r'past month', lambda m: "30d"),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, message)
            if match:
                return converter(match)
        return None

    def _extract_limit(self, message: str) -> int:
        """Extract limit/count from natural language"""
        patterns = [
            r'(?:top|first|show|limit)\s+(\d+)',
            r'(\d+)\s+(?:unique|different|distinct)',
            r'(?:maximum|max)\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return int(match.group(1))
        return None

    async def _oracle_fetcher_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute the appropriate Oracle log query"""
        try:
            if not state.intent:
                return {"oracle_data": {"type": "error", "message": "No intent analyzed"}}
            
            action = state.intent["action"]
            params = state.intent["params"]
            
            if action == "search_country":
                logs = await self.oracle_client.search_logs_by_country(params)
                return {"oracle_data": {"type": "logs", "data": logs, "count": len(logs)}}
            
            elif action == "search_ip":
                logs = await self.oracle_client.search_logs_by_ip(params)
                return {"oracle_data": {"type": "logs", "data": logs, "count": len(logs)}}
            
            elif action == "search_location":
                if "lat_min" not in params:
                    params.update({
                        "lat_min": 40.0, "lat_max": 45.0,
                        "lon_min": -80.0, "lon_max": -70.0
                    })
                logs = await self.oracle_client.search_logs_by_location(params)
                return {"oracle_data": {"type": "logs", "data": logs, "count": len(logs)}}
            
            else:  # analytics
                analytics = await self.oracle_client.get_traffic_analytics(params)
                return {"oracle_data": {"type": "analytics", "data": analytics}}
                
        except Exception as e:
            return {"oracle_data": {"type": "error", "message": str(e)}}

    def _generate_oracle_response(self, state: AgentState, chat_history: List[Dict[str, str]]) -> str:
        """Generate response using Gemini with Oracle data"""
        
        # Prepare context based on Oracle data
        oracle_context = ""
        if state.oracle_data:
            if state.oracle_data["type"] == "logs":
                oracle_context = f"""
    COMPLETE ORACLE CLOUD LOG DATASET ACCESS:
    - You have FULL ACCESS to the entire Oracle Cloud logs database
    - Current query returned {state.oracle_data['count']} matching log entries
    - This represents ALL entries matching the user's criteria (not a sample)
    - You can query any time range, IP address, country, or location from the complete dataset
    - The data shown below is the COMPLETE result set for this specific query

    Log entries found ({state.oracle_data['count']} total):
    """
                # Show ALL entries instead of limiting to 10
                for i, log in enumerate(state.oracle_data["data"]):
                    oracle_context += f"- {log.timestamp}: {log.ip} from {log.city}, {log.country} ({log.isp}) via {log.protocol}\n"
            
            elif state.oracle_data["type"] == "analytics":
                analytics = state.oracle_data["data"]
                oracle_context = f"""
    COMPLETE ORACLE CLOUD TRAFFIC ANALYTICS:
    - You have FULL ACCESS to analyze the entire Oracle Cloud logs database
    - These analytics represent the COMPLETE dataset for the specified time range
    - Total requests in dataset: {analytics.get('total_requests', 0)}
    - Unique IP addresses: {analytics.get('unique_ips', 0)}
    - Unique countries: {analytics.get('unique_countries', 0)}
    - Time range analyzed: {analytics.get('time_range', 'N/A')}
    - This is NOT a sample - this is the complete traffic data
    """
                
                if analytics.get('top_ip'):
                    oracle_context += f"\nComplete Top IP Addresses (from entire dataset):\n"
                    # Show ALL IP addresses instead of limiting to 15
                    for ip_data in analytics['top_ip']:
                        oracle_context += f"- {ip_data['name']}: {ip_data['count']} requests\n"
                
                if analytics.get('top_country'):
                    oracle_context += f"\nComplete Country Distribution: {analytics.get('top_country', [])}\n"
                
                oracle_context += f"Complete Protocol Distribution: {analytics.get('protocol_distribution', {})}\n"

        # Generate response with LLM
        prompt = f"""You are an Oracle Cloud logs analyst with COMPLETE DATABASE ACCESS. 

    IMPORTANT: You have full access to the entire Oracle Cloud logs database through the Oracle client API. 
    The data provided below represents the COMPLETE results for the user's query, not a sample or subset.
    Do not suggest that you have limited access or that this is partial data.

    Conversation history:
    {self._format_history(chat_history)}

    Current query: {state.query}
    System Context: {state.context}
    Analysis: {state.analysis}

    {oracle_context}

    INSTRUCTIONS:
    - Treat the provided data as the COMPLETE dataset for the user's query
    - Provide definitive answers based on this complete data
    - Do not suggest limitations or mention that you might not have access to all data
    - Give specific insights, patterns, and concrete answers
    - If asked about trends or comparisons, base them on the complete dataset you have access to
    """
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}\n\nRaw data summary:\n{oracle_context}"


    def _format_history(self, chat_history):
        formatted = ""
        for msg in chat_history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            else:
                formatted += f"Assistant: {msg['content']}\n"
        return formatted.strip()

    def _router_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """You are a query router that specializes in Oracle Cloud logs analysis. 
        Analyze the user's query to determine if it's related to log analysis, traffic analytics, 
        IP tracking, geographic analysis, or general conversation."""
        messages = [SystemMessage(content=system_msg)] + lc_messages
        response = self.llm.invoke(messages)
        return {
            "context": response.content,
            "iteration": state.iteration + 1
        }

    def _analyzer_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """Analyze the query and available context, including any Oracle log data. 
        Determine if additional research is needed or if you can provide a direct response."""
        
        context_info = f"Context: {state.context}"
        if state.oracle_data:
            context_info += f"\nOracle Data Available: {state.oracle_data['type']}"
        
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=context_info)
        ]
        response = self.analyzer.invoke(messages)
        analysis = response.content
        
        # If we have Oracle data, we can respond directly
        if state.oracle_data and state.oracle_data.get("type") != "error":
            next_action = "respond"
        elif "research" in analysis.lower() or "more information" in analysis.lower():
            next_action = "research"
        else:
            next_action = "respond"
            
        return {
            "analysis": analysis,
            "next_action": next_action
        }

    def _researcher_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """You are a research assistant. Gather relevant information to help answer queries 
        that are not related to Oracle Cloud logs (those are handled separately)."""
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=f"Analysis: {state.analysis}")
        ]
        response = self.llm.invoke(messages)
        updated_context = f"{state.context}\n\nResearch: {response.content}"
        return {"context": updated_context}

    def _validator_node(self, state: AgentState, lc_messages: List, response: str) -> Dict[str, Any]:
        system_msg = """Evaluate if the response adequately answers the query. 
        For Oracle logs queries, check if the data analysis is comprehensive and accurate."""
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=f"Response: {response}")
        ]
        validation_response = self.analyzer.invoke(messages)
        validation = validation_response.content
        return {"context": f"{state.context}\n\nValidation: {validation}"}
    

# --- Nord Theme CSS with Centered Layout ---
nord_theme_css = """
<style>
    /* Nord Color Palette */
    :root {
        --nord0: #2e3440;
        --nord1: #3b4252;
        --nord2: #434c5e;
        --nord3: #4c566a;
        --nord4: #d8dee9;
        --nord5: #e5e9f0;
        --nord6: #eceff4;
        --nord7: #8fbcbb;
        --nord8: #88c0d0;
        --nord9: #81a1c1;
        --nord10: #5e81ac;
        --nord11: #bf616a;
        --nord12: #d08770;
        --nord13: #ebcb8b;
        --nord14: #a3be8c;
        --nord15: #b48ead;
    }

    /* Main app background */
    .stApp {
        background-color: var(--nord0) !important;
        color: var(--nord4) !important;
    }

    /* Center the main content and limit width */
    .main .block-container {
        max-width: 800px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: var(--nord1) !important;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: var(--nord1) !important;
        border: 1px solid var(--nord2) !important;
        border-radius: 10px !important;
        max-width: 100% !important;
    }

    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: var(--nord10) !important;
        color: var(--nord6) !important;
    }

    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: var(--nord2) !important;
        color: var(--nord4) !important;
    }

    /* Chat input container */
    .stChatInput {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* Chat input */
    .stChatInput > div > div > input {
        background-color: var(--nord1) !important;
        color: var(--nord4) !important;
        border: 1px solid var(--nord3) !important;
        border-radius: 10px !important;
    }

    /* Text elements */
    .stMarkdown {
        color: var(--nord4) !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--nord8) !important;
    }

    /* Code blocks */
    .stCode {
        background-color: var(--nord1) !important;
        color: var(--nord13) !important;
        border: 1px solid var(--nord2) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--nord10) !important;
        color: var(--nord6) !important;
        border: none !important;
        border-radius: 5px !important;
    }

    .stButton > button:hover {
        background-color: var(--nord9) !important;
    }

    /* Links */
    a {
        color: var(--nord8) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--nord1) !important;
        color: var(--nord4) !important;
    }

    /* Metrics */
    .metric-container {
        background-color: var(--nord1) !important;
        border: 1px solid var(--nord2) !important;
        border-radius: 5px !important;
    }

    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background-color: var(--nord14) !important;
        color: var(--nord0) !important;
    }

    .stInfo {
        background-color: var(--nord8) !important;
        color: var(--nord0) !important;
    }

    .stWarning {
        background-color: var(--nord13) !important;
        color: var(--nord0) !important;
    }

    .stError {
        background-color: var(--nord11) !important;
        color: var(--nord6) !important;
    }

    /* Title centering */
    .main h1 {
        text-align: center !important;
    }
</style>
"""


# Update the Streamlit UI section
st.set_page_config(page_title="Oracle Logs Graph AI Agent", page_icon="📊")

# Apply Nord theme (keep your existing CSS)
st.markdown(nord_theme_css, unsafe_allow_html=True)

st.title("📊 Oracle Logs Graph AI Agent")
st.markdown("*LangGraph + Gemini + Oracle Cloud Logs Analytics*")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Configuration options in sidebar
with st.sidebar:
    st.header("Configuration")
    use_ollama = st.checkbox("Use Ollama", value=True, help="Uncheck to use Gemini")
    
    if use_ollama:
        ollama_model = st.selectbox(
            "Ollama Model",
            ["mistral-nemo:12b", "deepseek-r1:14b", "codellama:13b", "deepseek-coder:6.7b", "granite-code:8b"],
            index=0
        )
        st.info("Make sure Ollama is running on localhost:11434")
    else:
        st.info("Using Google Gemini (requires API key)")

# Initialize agent with selected configuration
if use_ollama:
    agent = OracleGraphAIAgent(use_ollama=True, ollama_model=ollama_model)
else:
    agent = OracleGraphAIAgent(use_ollama=False)


# Display existing chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Ask about Oracle logs or anything else...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Agent reasoning and streaming output
    with st.chat_message("assistant"):
        response = agent.run_with_reasoning(st.session_state.chat_history, st)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
