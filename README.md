# ReAct Agent with Time Travel and Human-in-the-Loop Review

---

## Overview

This project implements a ReAct (Reasoning + Acting) agent using LangGraph and LangChain, enhanced with:

- Custom tools integration
- Human-in-the-loop (HITL) for sensitive content review
- Time travel support via state checkpointing and resuming
- A modular, graph-based flow for agent logic control
  
---

![Alt text](agent_image.png)


## Features

- **ReAct Agent**: Combines reasoning and tool usage iteratively.
- **Human Review**: Sensitive or flagged outputs are routed to human review.
- **Tool Integration**: Custom tools invoked automatically by the agent.
- **Checkpointing & Time Travel**: Save intermediate states and resume with optional modifications.
- **Configurable via `Configuration` class** with system prompts, model choice, and tools.

---

## Project Structure

- **agent.py** (main logic, graph setup, nodes)
- **tools.py** (custom tool implementations)
- **state.py** (State and InputState data models)
- **configuration.py** (configuration loader and schema)
- **utils.py** (utility functions like loading the model)
- **README.md** (this file)

---

## Complete Agent Flow Explained

1. **Start:** Input state is created with user input.
2. **call_model Node:**
   - Loads the configured chat model and tools.
   - Sends system prompt + conversation history to the model.
   - Receives AIMessage response (may include tool calls).
3. **Routing:**
   - If AIMessage contains sensitive info (e.g. "suicide", "password"), route to `human_review`.
   - If tool calls present, route to `tools`.
   - Otherwise, end the graph (`__end__`).
4. **tools Node:**
   - Executes requested tools and appends their results.
   - Loops back to `call_model` with updated state.
5. **human_review Node:**
   - Presents flagged message to human for manual intervention.
   - Ends after review.
6. **Checkpointing & Time Travel:**
   - Before each node execution, the current state is serialized and saved as a checkpoint.
   - Agent can resume from any checkpoint ID, optionally modifying state.
   - Allows debugging, error recovery, and fine-tuning interaction.

---

## Summary Table of Components and Their Roles

| Component / Function         | Purpose / Description                                                                                      | Notes / Key Details                                |
|-----------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `call_model` node            | Calls the chat model with system prompt and conversation history; handles AIMessage responses              | Binds custom tools via `bind_tools(TOOLS)`         |
| `human_review` node          | Handles messages flagged as sensitive; routes output to human for manual review                            | Uses keyword filter for sensitive content detection|
| `ToolNode` (`tools`)         | Executes custom tools requested by the model, e.g., API calls or database queries                          | Loops back to `call_model` after tool execution    |
| `route_model_output`         | Conditional routing function to determine next node based on message content                               | Routes to `tools`, `human_review`, or `__end__`   |
| `contains_sensitive_info()` | Checks if AI message contains sensitive keywords requiring human review                                    | Simple keyword list; extendable for NLP-based detection|
| `State` class                | Represents the agent's current state including input and conversation messages                             | Must implement `to_dict()` and `from_dict()` for checkpointing |
| Checkpoint storage (`CHECKPOINTS`) | Stores serialized states for time travel and resuming agent runs                                  | In-memory dict for demo; replace with DB in prod  |
| `serialize_state()`          | Serializes `State` to JSON string for checkpointing                                                       | Uses `State.to_dict()`                              |
| `deserialize_state()`        | Loads JSON string back into `State` object                                                                | Uses `State.from_dict()`                            |
| `run_agent_with_time_travel` | Runs the agent from scratch or from a saved checkpoint with optional modifications                        | Saves checkpoints before each step                  |
| `Configuration` class        | Loads system prompt template, model selection, and tool configuration                                      | Supports dynamic system prompts with timestamps    |



## Problems Faced & How They Were Solved

### 1. Setting up the Project with LangSmith Studio  
**Problem:** Integrating LangGraph flows with LangSmith Studio’s visualization and debugging was tricky due to lack of direct tooling.  

**Solution:** Used manual graph compilation and explicit checkpoint serialization for visualization. Planned future integration with LangSmith’s tracing APIs to improve debugging and tracing support.

---

### 2. Integration of Custom Tools  
**Problem:** Custom tool calls needed to be bound properly to the model’s tool system and routed back into the graph loop to enable multi-step reasoning and tool usage.  

**Solution:** Implemented `ToolNode` from `langgraph.prebuilt` and ensured all tools were registered and properly bound in the `call_model` node using `bind_tools(TOOLS)`. This allowed seamless invocation of tools during the agent run.

---

### 3. State Serialization and Time Travel  
**Problem:** Serializing complex `State` objects containing nested messages and tool call data for checkpointing and resuming was challenging.  

**Solution:** Added `to_dict()` and `from_dict()` methods in the `State` and message classes to convert the state into JSON-serializable dicts. Used JSON as the checkpoint serialization format to enable easy saving, loading, and modifying of state snapshots.

---

### 4. Handling Sensitive Content with Human-in-the-Loop  
**Problem:** Detecting sensitive or potentially harmful content in AI output and properly routing it to a human for review was error-prone and required careful filtering.  

**Solution:** Created a simple keyword-based filter in `contains_sensitive_info()` to detect flagged terms like "suicide" or "password". Combined with a routing function `route_model_output()` to redirect flagged messages to the `human_review` node, enabling human intervention and safer outputs.

---

## Additional Tips

- Ensure your `State` class correctly implements `to_dict()` and `from_dict()` to enable reliable checkpointing.
- Extend `contains_sensitive_info()` with more advanced NLP techniques or regex filters to improve sensitive content detection.
- For production, store checkpoints in persistent storage such as databases or filesystem instead of in-memory dictionaries.
- Utilize LangSmith Studio or other graph visualization tools for debugging and tracing agent execution flows.

