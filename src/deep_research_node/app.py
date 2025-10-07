import os
"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from state_research import ResearcherState, ResearcherOutputState
from utils import tavily_search, get_today_str, think_tool, ask_gpt_5
# from prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from PromptManager import PromptManager

# ===== CONFIGURATION =====

# Set up tools and model binding
tools = [
    tavily_search,
    think_tool,
    ask_gpt_5
]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize models
model = init_chat_model(model="openai:gpt-4.1")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="openai:gpt-4.1-mini")
compress_model = init_chat_model(model="openai:gpt-4.1", max_tokens=32000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000
qa_model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", max_tokens=64000)

# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    prompt_manager = PromptManager()
    research_agent_prompt = prompt_manager.get_prompt("research_agent_prompt_user", "v1.0.0").render(date=get_today_str())
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    prompt_manager = PromptManager()
    compress_research_system_prompt = prompt_manager.get_prompt("compress_research_prompt_system", "v1.0.0")
    system_message = compress_research_system_prompt.render(date=get_today_str())
    compress_research_human_message = prompt_manager.get_prompt("compress_research_prompt_user", "v1.0.0").render(research_topic=state["research_topic"])
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

def assess_quality(state: ResearcherState) -> dict:
    """Assess the quality of the research findings.

    Takes the compressed research and performs a quality assurance check
    using the geologic QA prompt to evaluate the report.
    """
    prompt_manager = PromptManager()
    geologic_qa_prompt = prompt_manager.get_prompt("geologic_qa_prompt_user", "v1.0.0")
    qa_prompt_content = geologic_qa_prompt.render(research_report=state["compressed_research"])
    
    messages = [HumanMessage(content=qa_prompt_content)]
    response = qa_model.invoke(messages)
    
    return {
        "qa_report": str(response.content)
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)
agent_builder.add_node("assess_quality", assess_quality)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", "assess_quality") # Go to quality assessment
agent_builder.add_edge("assess_quality", END)

# Compile the agent
researcher_agent = agent_builder.compile()

if __name__ == "__main__":
    # Basic runnable example showing how to create a prompt and invoke the graph
    from langchain_core.messages import HumanMessage
    
    prompt_manager = PromptManager()
    geologic_research_prompt_user = prompt_manager.get_prompt("geologic_research_prompt_user", "v1.0.0")
    latitude = '34.134097'
    longitude = '-81.638175'
    prior_land_use = 'agricultural'
    site_name = 'Saluda'

    # user_prompt = os.environ.get("RESEARCH_PROMPT") or "What are the key differences between CRDTs and Operational Transform?"
    user_prompt = geologic_research_prompt_user.render(latitude=latitude, longitude=longitude, prior_land_use=prior_land_use, site_name=site_name)
    initial_state = {
        "researcher_messages": [HumanMessage(content=user_prompt)],
        "tool_call_iterations": 0,
        "research_topic": user_prompt,
        "compressed_research": "",
        "raw_notes": [],
        "qa_report": "",
    }

    print("Starting research...\n")
    result = researcher_agent.invoke(initial_state)

    print("=== Compressed Research ===\n")
    print(result.get("compressed_research", ""))
    print("\n=== QA Report ===\n")
    print(result.get("qa_report", ""))
    # print("\n=== Raw Notes (truncated) ===\n")
    # raw_notes = "\n\n".join(result.get("raw_notes", [])).strip()
    # print(raw_notes[:2000] + ("..." if len(raw_notes) > 2000 else ""))
    print('available keys: ', result.keys())
