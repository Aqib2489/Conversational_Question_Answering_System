# Import necessary libraries
import streamlit as st
import base64
import io
from PIL import Image as PILImage
from huggingface_hub import login
from byaldi import RAGMultiModalModel
import google.generativeai as genai

# Configure Google Gemini API
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Load the RAG multi-modal model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

# Specify the index path where the index was saved
index_path = "/home/mohammadaqib/Desktop/project/research/Multi-Modal-RAG/Colpali/BCC"
RAG = RAGMultiModalModel.from_index(index_path)

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit app title
st.title("Conversational Multi-Modal RAG Chatbot")

# Input field for user query
query = st.text_input("Enter your question:")

# Handle the conversation on button click
if st.button("Send") and query:
    # Query the RAG model with the user's question
    results = RAG.search(query, k=3)
    
    # Check for images in results
    image_list = []
    num_results = len(results)
    for i in range(min(3, num_results)):
        try:
            image_bytes = base64.b64decode(results[i].base64)
            image = PILImage.open(io.BytesIO(image_bytes))
            image_list.append(image)
        except AttributeError:
            st.write(f"Result {i} does not contain an 'base64' attribute.")
    
    # Process images if available and merge them
    if image_list:
        total_width = sum(img.width for img in image_list)
        max_height = max(img.height for img in image_list)
        merged_image = PILImage.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in image_list:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
        merged_image.save('merged_image.jpg')
        img = PILImage.open('merged_image.jpg')

        # Generate response with Google Gemini
        response = model.generate_content([f'Answer to the question asked using the image. Also mention the reference from image to support your answer. Example, Table Number or Statement number or any metadata. Question: {query}', img], stream=True)
        response.resolve()
        text = response.text
    else:
        text = "No images were found in the search results."

    # Append user query and AI response to conversation history
    st.session_state.conversation_history.append({"user": query, "bot": text})
    
    # Clear the input field after submission
    st.text_input("Enter your question:", value="", key="new")

# Display the conversation history
for entry in st.session_state.conversation_history:
    st.markdown(f"**User:** {entry['user']}")
    st.markdown(f"**Bot:** {entry['bot']}")
    st.write("---")  # Divider between messages

