import random
from typing import List

from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, MessageGraph, StateGraph
from langgraph.graph.message import add_messages

from chains import revisor_chain, first_responder_chain, pydantic_tool_parser, PydanticToolsParser
from execute_tools import execute_tool
from schema import ReviseAnswer


class OverallState(TypedDict):
    messages:Annotated[list, add_messages]
    parsed_output:object

graph = StateGraph(OverallState)

MAX_ITERATIONS = 6

def draft_node(State:OverallState):
    response = first_responder_chain.invoke({"messages":State["messages"]})
    parsed_response = pydantic_tool_parser.invoke(response)
    return {"messages":[response], "parsed_output":[parsed_response]}

def revisor_node(State:OverallState):
    response = revisor_chain.invoke({"messages":State["messages"]})
    wait_dict = {1:"Thinking...", 2:"Reframing the response...", 3:"Analyzing the output...", 4:"Wait, Let me search internet on this..."}
    wait_dict_index = random.randint(1,4)
    print(wait_dict[wait_dict_index])
    parsed_response = PydanticToolsParser(tools=[ReviseAnswer]).invoke(response)
    return {"messages":[response],"parsed_output":[parsed_response]}

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

user_input = input("What topic to research today :: ")

response = app.invoke({"messages":[HumanMessage(content=user_input)]})

print("Parsed response :: ", response["parsed_output"][-1])

print("Final Response:: ", response["messages"][-1].tool_calls[0]["args"]["answer"])