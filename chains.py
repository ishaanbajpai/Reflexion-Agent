from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

pydantic_tool_parser = PydanticToolsParser(tools=[AnswerQuestion])

#Actor Agent Prompt
actor_agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     
     """You are expert AI researcher.
     Current time: {time}
     
     1. {first_instruction}
     2. Reflect and critique your answer. Be sever to maximize improvements.
     3. After the relfection, **list 1-3 search queries separately** for 
     reseaching improvements. Do not include them inside reflection
     """),
     MessagesPlaceholder("messages"),
     ("system", "Answer the user's question above using the required format")
]).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_agent_prompt.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_agent_prompt.partial(first_instruction=revise_instructions) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# response = first_responder_chain.invoke({
#     "messages": [HumanMessage(content="Write me a blogpost on how small businesses can leverage AI to grow")]
# })