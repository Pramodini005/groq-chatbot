import os
import streamlit as st
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph & Groq Chatbot")

# --- API Key Setup ---
# Priority: Streamlit secrets first, then environment variables, then sidebar input
groq_api_key = (
    st.secrets.get("GROQ_API_KEY") 
    or os.environ.get("GROQ_API_KEY")
)

with st.sidebar:
    st.header("Configuration")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        st.info("Tip: You can set GROQ_API_KEY in `.streamlit/secrets.toml` or environment variables to skip entering it here.")
    
    selected_model = st.selectbox(
        "Select Model",
        options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

# --- LangGraph Setup ---
class AgentState(TypedDict):
    messages: List[BaseMessage]

@st.cache_resource
def get_agent(api_key: str, model_name: str):
    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key
    )

    def process(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(AgentState)
    graph.add_node("process_node", process)
    graph.add_edge(START, "process_node")
    graph.add_edge("process_node", END)
    return graph.compile()

try:
    agent = get_agent(groq_api_key, selected_model)
except Exception as e:
    st.error(f"Failed to initialize LLM client: {e}")
    st.stop()

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# --- Handle User Input ---
if user_input := st.chat_input("Type your message..."):
    # Render user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Append user message to state
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Invoke agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": st.session_state.messages})
            # Get the newly appended AI message
            ai_response = result["messages"][-1]
            st.write(ai_response.content)
            
            # Synchronize Streamlit session state with graph state
            st.session_state.messages = result["messages"]
