import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import time
import base64
import subprocess


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_Tracing_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Function to pull the models
def pull_models(model_list):
    for model in model_list:
        print(f"Pulling model: {model}")
        subprocess.run(["ollama", "pull", model], check=True)

model_options = ["llama3"] 


# Encode image to Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

#path of the image for frontent app
image_path = r"C:\Users\arifa\Downloads\olama.jpeg"  
img_base64 = get_base64_of_bin_file(image_path)

# css for background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h1.title {{
    font-size: 3.5rem;
    font-weight: bold;
    color: #FFD700;
    text-shadow: 2px 2px 4px #000000;
    margin-bottom: 20px;
    text-align: center;
}}

p.designer {{
    font-size: 1.2rem;
    font-weight: normal;
    color: #ADD8E6;
    text-align: center;
    margin-top: -10px;
}}

label.input-label {{
    font-size: 1.5rem;
    font-weight: bold;
    color: #FFFFFF;
    background-color: #000000;
    padding: 5px 10px;
    border-radius: 5px;
    display: inline-block;
}}

textarea.stTextArea {{
    font-size: 1.1rem;
}}

.response-container {{
    background-color: rgba(0, 0, 0, 0.7);
    border: 2px solid #FFD700;
    padding: 20px;
    border-radius: 10px;
    max-height: 400px;
    overflow-y: auto;
    color: #FFFFFF;
}}
</style>
"""

#custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

#title
st.markdown('<h1 class="title">LlamaTalk using Ollama</h1>', unsafe_allow_html=True)

#Highlight the name
st.markdown('<p class="designer">Designed by <b>Arif Ansari</b></p>', unsafe_allow_html=True)

# Dropdown for model selection but only i used the llama3 latest model.
selected_model = st.selectbox("Select a model to use:", model_options)

# Prompt Template
default_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "Hello, how are you?"),
        ("ai", "I'm doing well, thanks!"),
        ("user", "That's good to hear."),
        ("user", "Question: {question}")
    ]
)

st.markdown('<label class="input-label">Enter your prompt:</label>', unsafe_allow_html=True)
prompt_input = st.text_area("", height=150)

chain = None
response_container = st.empty()


if selected_model and prompt_input:
    llm = Ollama(model=selected_model)  # Use the latest model or select "llama3:
    chain = LLMChain(llm=llm, prompt=default_prompt, output_parser=StrOutputParser())

# Now Generate the response
if st.button("Generate Response"):
    if chain:
        with st.spinner("Generating response..."):
            full_response = ""
            try:
                for chunk in llm.stream(prompt_input):
                    if isinstance(chunk, dict) and "text" in chunk:
                        full_response += chunk["text"]
                    elif isinstance(chunk, str):
                        full_response += chunk
                    else:
                        st.error("Unexpected chunk type encountered.")
                        break
                    
                    response_html = f"""
                    <div class="response-container">
                        <p>{full_response}</p>
                    </div>
                    """
                    response_container.markdown(response_html, unsafe_allow_html=True)
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.write("Please select a model and enter a prompt to generate a response.")
