from datetime import UTC, datetime
from typing import Dict, List, Literal, Optional, cast
import json

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


# --- Your existing nodes ---

async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response]}


async def human_review(state: State) -> Dict[str, List[AIMessage]]:
    last_message = state.messages[-1]
    return {"messages": [last_message]}


def contains_sensitive_info(message: AIMessage) -> bool:
    content = (message.content or "").lower()
    sensitive_keywords = ["suicide", "kill myself", "password", "credentials"]
    return any(keyword in content for keyword in sensitive_keywords)


def route_model_output(state: State) -> Literal["__end__", "tools", "human_review"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    if contains_sensitive_info(last_message):
        return "human_review"

    if last_message.tool_calls:
        return "tools"

    return "__end__"


# --- Build your graph ---

builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(human_review)

builder.add_edge("__start__", "call_model")

builder.add_conditional_edges("call_model", route_model_output)

builder.add_edge("tools", "call_model")

builder.add_edge("human_review", "__end__")

graph = builder.compile(name="ReAct Agent")


# --- Time travel support ---


# In-memory checkpoint storage (in production, use DB or file system)
CHECKPOINTS = {}


def serialize_state(state: State) -> str:
    """
    Serialize the state object to JSON for checkpointing.
    Implement this method according to your State class structure.
    """
    # This example assumes your State can be converted to dict or JSON
    # You need to adapt this based on your actual State implementation
    return json.dumps(state.to_dict())  # You need to implement `to_dict()` in State


def deserialize_state(serialized: str) -> State:
    """
    Deserialize JSON string back to State object.
    """
    data = json.loads(serialized)
    return State.from_dict(data)  # You need to implement `from_dict()` in State


async def run_agent_with_time_travel(
    initial_input: InputState,
    resume_checkpoint_id: Optional[str] = None,
    modifications: Optional[Dict] = None,
) -> State:
    """
    Run the agent either from scratch or resume from a checkpoint with optional modifications.
    """

    if resume_checkpoint_id is not None:
        # Load the checkpoint state
        serialized_state = CHECKPOINTS.get(resume_checkpoint_id)
        if serialized_state is None:
            raise ValueError(f"Checkpoint {resume_checkpoint_id} not found")

        state = deserialize_state(serialized_state)

        # Apply modifications if any (e.g., modify messages or other fields)
        if modifications:
            for key, value in modifications.items():
                setattr(state, key, value)

    else:
        # Start from initial input state
        state = State(input=initial_input)

    # Run the graph until terminal state
    while True:
        # Save checkpoint before executing node (so you can come back here)
        checkpoint_id = f"ckpt_{len(CHECKPOINTS)}"
        CHECKPOINTS[checkpoint_id] = serialize_state(state)
        print(f"Checkpoint saved: {checkpoint_id}")

        # Execute the next step in the graph
        output = await graph.execute(state)

        # Update the state with new messages
        if "messages" in output:
            state.messages.extend(output["messages"])

        # Check if graph reached the end
        if graph.is_terminal(state):
            break

    # Save the final checkpoint as well
    final_ckpt_id = f"ckpt_{len(CHECKPOINTS)}"
    CHECKPOINTS[final_ckpt_id] = serialize_state(state)
    print(f"Final checkpoint saved: {final_ckpt_id}")

    return state


def list_checkpoints() -> List[str]:
    """
    List all checkpoint IDs stored.
    """
    return list(CHECKPOINTS.keys())
