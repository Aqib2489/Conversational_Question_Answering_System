import streamlit as st
import base64
import os
import PIL.Image
import io
from huggingface_hub import login
from byaldi import RAGMultiModalModel
from PIL import Image as PILImage
import google.generativeai as genai

# Configuration for Google Generative AI
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Initialize RAG model
index_path = "/home/mohammadaqib/Desktop/project/research/Multi-Modal-RAG/Colpali/BCC"
RAG = RAGMultiModalModel.from_index(index_path)

# Streamlit page setup
st.set_page_config(page_title="Multi-Modal RAG Chatbot", layout="wide")
st.title("🔍 Multi-Modal RAG Chatbot")

# Sidebar for file upload (if needed)
st.sidebar.header("Settings")
query = st.sidebar.text_input("Enter your query", value="What are the locations where flashing needs to be installed?")

# Create a container for chatbot
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Ask your question:", query)

# On button click, perform the search and generate response
if st.button("Submit"):
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        # Query the RAG model
        results = RAG.search(user_input, k=3)
        num_results = len(results)
        st.write(f"Number of results available: {num_results}")

        # Prepare images from the search results
        image_list = []
        for i in range(min(3, num_results)):
            try:
                image_bytes = base64.b64decode(results[i].base64)
                image = PILImage.open(io.BytesIO(image_bytes))
                image_list.append(image)
            except AttributeError:
                st.write(f"Result {i} does not contain an 'base64' attribute.")

        # Display merged images if any
        if image_list:
            # Calculate dimensions for merged image
            total_width = sum(img.width for img in image_list)
            max_height = max(img.height for img in image_list)
            merged_image = PILImage.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in image_list:
                merged_image.paste(img, (x_offset, 0))
                x_offset += img.width

            # Display merged image in Streamlit
            st.image(merged_image, caption="Merged Image from Search Results")

            # Pass the merged image to the generative model
            img_byte_arr = io.BytesIO()
            merged_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Generate content using Gemini
            response = model.generate_content([f"Answer the question using the image. Mention references like table or statement numbers. Question: {user_input}", merged_image], stream=True)
            response.resolve()
            answer = response.text

            # Append answer to chatbot history
            st.session_state.history.append({"role": "bot", "content": answer})
        else:
            st.write("No images were found in the search results.")
            st.session_state.history.append({"role": "bot", "content": "No relevant images were found."})

# Display the conversation history
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"**User**: {chat['content']}")
    else:
        st.markdown(f"**Bot**: {chat['content']}")

