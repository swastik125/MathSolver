
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain , LLMChain
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key= os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")

st.set_page_config(page_title="Text to Math Problem Solver",page_icon="🦜️")
st.title("Text to Math Problems Solver")
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    

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


prompt="""
Your a agent tasked for solving users mathematical question . Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)
chain=LLMChain(llm=llm,prompt=prompt_template)


reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and resoning question."
)
## Combine all the tools into chain


##initialize the agents
assistent_agent=initialize_agent(
    tools=[wikipidia_tool,calculator,reasoning_tool,python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)
if "messages" not in st.session_state:
    st.session_state["messages"]= [
        {"role":"assistant", "content":"Hi, i am a Math chatbot who can answer all your math question"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

##function to generate the response


##Start the interaction
question=st.text_area("Enter your question"," ")
if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistent_agent.run(st.session_state.messages)
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write("### Response: ")
            st.success(response)

    else:
        st.warning("Please enter the question")