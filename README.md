## deep-research-simple

An intentionally small example showing how to build an iterative research agent that calls a search tool and a "think tool" in a loop. The agent searches the web (via Tavily), reflects using a `think_tool` to decide whether the question is answered, and either stops or searches again. This repository focuses on clarity over features.

### Features
- **Iterative search loop** using `langgraph` to wire LLM steps and tool calls.
- **Web search** via `tavily_search` with optional raw-content summarization.
- **Deliberate reflection** via `think_tool` after each search to assess gaps and next steps.
- **Compression step** that cleans up and consolidates gathered findings.

### Requirements
- Python 3.10+
- Accounts and API keys for:
  - **OpenAI** (models used in this example)
  - **Tavily** (web search)

Set the following environment variables:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Optionally:
- `RESEARCH_PROMPT` (to override the default research question when running)

### Installation
```bash
pip install -r requirements.txt
```

Then export your API keys (macOS/Linux examples):
```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
# Optional: customize the question
export RESEARCH_PROMPT="What are the tradeoffs between vector databases and plain Postgres for RAG?"
```

### How it works
At a high level, the agent is a small state machine defined in `app.py` using `langgraph`:
1. **Create prompt/state**: A `HumanMessage` seed is placed in the `researcher_messages` state along with metadata (see `if __name__ == "__main__"` in `app.py`).
2. **LLM decides next action** (`llm_call`): The LLM, bound to tools, decides whether to call tools.
3. **Tool execution** (`tool_node`): If tool calls are present, they are executed. In this example:
   - `tavily_search` performs a Tavily web search, deduplicates results, optionally summarizes raw content, and returns a structured string of sources.
   - `think_tool` records a structured reflection: what was learned, what’s missing, and whether to continue.
4. **Routing** (`should_continue`): If the latest LLM message still has tool calls to make, the loop continues; otherwise it proceeds to compression.
5. **Compression** (`compress_research`): The gathered research content is cleaned and consolidated into `compressed_research` while omitting the internal reflections.

The critical behavior is the interplay between `tavily_search` and `think_tool`: after each search, the agent uses `think_tool` to explicitly assess whether the user’s question is adequately answered. If not, the loop continues with more targeted searches.

### Key files
- `app.py`: Builds the research graph, routing, and includes a runnable example under the main guard.
- `utils.py`: Implements `tavily_search`, the summarization helpers, result deduplication/formatting, and the `think_tool`.
- `state_research.py`: Strongly-typed state and output schemas for the agent.
- `prompts.py`: Prompt templates for research behavior and compression.

### Running the example
After installing dependencies and exporting API keys:
```bash
python app.py
```

What happens:
- The script reads `RESEARCH_PROMPT` if set, otherwise uses a default question.
- It constructs the initial state with that prompt and invokes the compiled graph (`researcher_agent`).
- The agent will likely call `tavily_search` first, then `think_tool`. If the reflection indicates gaps, it searches again; otherwise it stops.
- Results are printed:
  - `Compressed Research`: a cleaned, consolidated view of the gathered findings
  - `Raw Notes` (truncated): tool outputs and AI messages for transparency

### Minimal usage pattern (what you’d typically write)
In `app.py` under the main guard we create a prompt and invoke the graph:
```python
from langchain_core.messages import HumanMessage

user_prompt = os.environ.get("RESEARCH_PROMPT") or "What are the key differences between CRDTs and Operational Transform?"

initial_state = {
    "researcher_messages": [HumanMessage(content=user_prompt)],
    "tool_call_iterations": 0,
    "research_topic": user_prompt,
    "compressed_research": "",
    "raw_notes": [],
}

result = researcher_agent.invoke(initial_state)
print(result["compressed_research"])  # final cleaned findings
```

### Notes on models and tools
- The example initializes OpenAI chat models (`gpt-4.1` and `gpt-4.1-mini`) via `langchain`.
- Tools are registered and bound to the model, allowing the LLM to decide which to call.
- Tavily search requires `TAVILY_API_KEY`; OpenAI usage requires `OPENAI_API_KEY`.

### Troubleshooting
- "Import could not be resolved" warnings usually indicate a missing dependency or environment mismatch. Ensure `pip install -r requirements.txt` ran successfully.
- Tavily errors typically mean the `TAVILY_API_KEY` is missing or invalid.
- Empty or low-quality results? Try a clearer `RESEARCH_PROMPT` or increase `max_results` inside `tavily_search`.

### License
MIT


