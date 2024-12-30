from langchain_community.llms import Ollama
import streamlit as st
import time
import base64

# Encode the image to Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Path to the local image
image_path = r"olama.jpeg"  #olama.jpeg  # Use raw string (r"") for Windows file paths
img_base64 = get_base64_of_bin_file(image_path)

# Inject custom CSS for the background and styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h1.title {{
    font-size: 3.5rem; /* Larger size for the title */
    font-weight: bold;
    color: #FFD700; /* Gold color for text on dark background */
    text-shadow: 2px 2px 4px #000000; /* Subtle shadow for visibility */
    margin-bottom: 20px;
    text-align: center;
}}

p.designer {{
    font-size: 1.2rem; /* Slightly smaller text for the designer */
    font-weight: normal;
    color: #ADD8E6; /* Light blue for contrast */
    text-align: center;
    margin-top: -10px;
}}

label.input-label {{
    font-size: 1.5rem; /* Bigger size for input label */
    font-weight: bold;
    color: #FFFFFF; /* White text */
    background-color: #000000; /* Black background for highlight */
    padding: 5px 10px;
    border-radius: 5px;
    display: inline-block;
}}

textarea.stTextArea {{
    font-size: 1.1rem; /* Slightly bigger font for text area */
}}

.response-container {{
    background-color: rgba(0, 0, 0, 0.7); /* Dark semi-transparent background */
    border: 2px solid #FFD700; /* Gold border */
    padding: 20px;
    border-radius: 10px;
    max-height: 400px; /* Optional: limit height */
    overflow-y: auto; /* Enable scrolling if content exceeds height */
    color: #FFFFFF; /* Response text color */
}}
</style>
"""

# Apply the custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add the app title
st.markdown('<h1 class="title">LlamaTalk using Ollama</h1>', unsafe_allow_html=True)

# Add the designer attribution
st.markdown('<p class="designer">Designed by <b>Arif Ansari</b></p>', unsafe_allow_html=True)

# Initialize the LLM
llm = Ollama(model="llama3")

# Input field for the prompt
st.markdown('<label class="input-label">Enter your prompt:</label>', unsafe_allow_html=True)
prompt = st.text_area("", height=150)

# Create the bounding box for responses
st.markdown("### **Chatbot Response**")
response_container = st.empty()  # Placeholder for chatbot response in the styled container

# Button to trigger generation
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            # Store the complete response progressively
            full_response = ""

            try:
                # Iterate over the generator and update the response container
                for chunk in llm.stream(prompt):
                    if isinstance(chunk, dict) and "text" in chunk:
                        full_response += chunk["text"]
                    elif isinstance(chunk, str):
                        full_response += chunk
                    else:
                        st.error("Unexpected chunk type encountered")
                        break

                    # Update the styled container with bolded text dynamically
                    response_html = f"""
                    <div class="response-container">
                        <p>{full_response}</p>
                    </div>
                    """
                    response_container.markdown(response_html, unsafe_allow_html=True)

                    # Simulate real-time streaming
                    time.sleep(0.1)

            except Exception as e:
                st.error(f"An error occurred: {e}")
