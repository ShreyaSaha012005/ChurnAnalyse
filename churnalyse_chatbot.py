import openai
import streamlit as st
import os
from dotenv import load_dotenv

# Load API key from .env file or environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not api_key:
    st.error("OpenAI API key is missing! Set it in a .env file or as an environment variable.")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Function to get chatbot response
def get_response(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Churnalyse Chatbot 🤖")
st.write("Ask me anything!")

user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input.strip():
        response = get_response(user_input)
        st.text_area("Bot:", value=response, height=100, disabled=True)
    else:
        st.warning("Please enter a message!")

