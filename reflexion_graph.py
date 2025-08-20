from typing import List

from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph, StateGraph
from langgraph.graph.message import add_messages

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tool

class OverallState(TypedDict):
    messages:Annotated[list, add_messages]

graph = StateGraph(OverallState)

MAX_ITERATIONS = 6

def draft_node(State:OverallState):
    response = first_responder_chain.invoke({"messages":State["messages"]})
    return {"messages":[response]}

def revisor_node(State:OverallState):
    response = revisor_chain.invoke({"messages":State["messages"]})
    print("revisor node print")
    return {"messages":[response]}

graph.add_node("draft", draft_node)
graph.add_node("execute_tool", execute_tool)
graph.add_node("revisor", revisor_node)

graph.add_edge("draft", "execute_tool")
graph.add_edge("execute_tool", "revisor")

def event_loop(State:OverallState):
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in State["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    else:
        return "execute_tool"

graph.add_conditional_edges("revisor", event_loop,
                            {
                                END:END,
                                "execute_tool":"execute_tool"
                            }
                            )
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke({"messages":"Write about how small business can leverage AI to grow"})

print("response :: ", response["messages"][-1])

print("Final Response:: ", response["messages"][-1].tool_calls[0]["args"]["answer"])