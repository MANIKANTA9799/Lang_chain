from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool 
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field 
from typing import List
import os 
load_dotenv()
class Source(BaseModel):
    """
    Schema for a source used by an agent 
    """
    url : str = Field(description="The Url of the source ")


class AgentResponse(BaseModel):
    """
    Schema for agent response with a answer and sources 
    """
    answer : str = Field(description="the agent's answer to the query ")
    sources : List[Source] = Field(default_factory=list,description="the list of sources to generate the answer")

# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query:str) :
#     """
#     Tool that searches over internet 
#     Args :
#         query : The query to search for 
#     Returns : 
#           The search result 
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

# here above we wrote a custom sercah query using tavuly but for that we need to mastert in llm working 
# so we tryst tavily with its default search to do the job 

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0
)
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools, response_format=AgentResponse)
def main():
   pass
   result = agent.invoke({
    "messages": [
        HumanMessage(content="What is the weather in Tokyo today")
    ]

})
   print(result)
if __name__ == "__main__":
    main()
