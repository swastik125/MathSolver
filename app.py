import speech_recognition as sr
import streamlit as st
from tool import  python_tool, calculator, wikipidia_tool
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain , LLMChain
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")

st.set_page_config(page_title="Text to Math Problem Solver",page_icon="🦜️")
st.title("Text to Math Problems Solver")

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



##function for voice input
def voice_to_text():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak now!")
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            st.info("⏳ Processing...")
            text = r.recognize_google(audio)  # Google Speech API
            return text
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")
        return None


##Start the interaction
choice = st.radio("Select input type:", ["Text", "Image", "Voice"])
question=""

if choice == "Text":
    question = st.text_area("Enter your math question:")

# -------- IMAGE MODE --------
elif choice == "Image":
    uploaded = st.file_uploader("Upload a math problem image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text from image..."):
            extracted_text = pytesseract.image_to_string(img)
            question = extracted_text.strip()
            st.write("**Extracted Question:**", question)

# -------- VOICE MODE --------
elif choice == "Voice":
    if st.button("🎙️ Record Voice"):
        text = voice_to_text()
        if text:
            st.session_state["voice_question"] = text
            st.info(f"🗣️ You said: {text}")            

        else:
            st.warning("Couldn't recognize speech. Please try again.")
    question = st.session_state.get("voice_question", "")

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