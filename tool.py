from langchain.agents import Tool,initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain , LLMChain
from langchain_experimental.tools.python.tool import PythonREPLTool
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_api_key= os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")  

## Initialize Tools
wikipidia_wrapper=WikipediaAPIWrapper()
wikipidia_tool=Tool(
    name="Wikipedia",
    func=wikipidia_wrapper.run,
    description="A Tool for searching the Internet to find the Information" 
)
##Initialize the Math tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator= Tool(
    name="Calculator",
    func=math_chain.run,
    description="A Tool for answering math related questions. Only input mathematical expression"
)


python_tool = Tool(
    name="Python Executor",
    func=PythonREPLTool().run,
    description=(
        "A code interpreter that can perform advanced math. "
        "Use it when you need to run Python code, including NumPy, SymPy, or SciPy for algebra, statistics, calculus, or complex logic."
    )
)