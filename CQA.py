import base64
import os
import PIL.Image
from byaldi import RAGMultiModalModel
from PIL import Image as PILImage
import io
import textwrap

import google.generativeai as genai
import streamlit as st
from IPython.display import display, Image as DisplayImage  # Import display and rename Image to avoid conflict

# Function to initialize the Google API with the user's API key
def initialize_google_api(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Streamlit UI setup
st.title("Multi-Modal Query System")

# Prompt the user to input their Google API Key
google_api_key = st.text_input("Enter your Google API Key", type="password")

# Proceed only if API Key is entered
if google_api_key:
    # Initialize Google API and model
    model = initialize_google_api(google_api_key)
    
    # Load the RAG multi-modal model
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

    # Specify the index path where the index was saved during the first run
    index_path = "/home/mohammadaqib/Desktop/project/research/Multi-Modal-RAG/Colpali/BCC"
    RAG = RAGMultiModalModel.from_index(index_path)

    # Initialize conversation history
    conversation_history = []

    # Define functions for user interaction, image processing, and generating answers
    def get_user_input():
        """Prompt the user for input."""
        return st.text_input("Enter your question")

    def process_image_from_results(results):
        """Process images from the search results and merge them."""
        image_list = []
        for i in range(min(3, len(results))):
            try:
                # Ensure the result has a base64 attribute
                image_bytes = base64.b64decode(results[i].base64)
                image = PILImage.open(io.BytesIO(image_bytes))  # Open image directly from bytes
                image_list.append(image)
            except AttributeError:
                st.write(f"Result {i} does not contain a 'base64' attribute.")
        
        # Merge images if any
        if image_list:
            total_width = sum(img.width for img in image_list)
            max_height = max(img.height for img in image_list)

            merged_image = PILImage.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in image_list:
                merged_image.paste(img, (x_offset, 0))
                x_offset += img.width

            # Save the merged image
            merged_image.save('merged_image.jpg')
            return merged_image
        else:
            return None

    def generate_answer(query, image):
        """Generate an answer using the Gemini model and the merged image."""
        response = model.generate_content([f'Answer to the question asked using the image. Also mention the reference from image to support your answer. Example, Table Number or Statement number or any metadata. Question: {query}', image], stream=True)
        response.resolve()
        return response.text

    def classify_system_question(query):
        """Check if the question is related to the system itself."""
        response = model.generate_content([f"Determine if the question is about the system itself, like 'Who are you?' or 'What can you do?' or 'Introduce yourself' . Answer with 'yes' or 'no'. Question: {query}"], stream=True)
        response.resolve()
        return response.text.strip().lower() == "yes"

    def classify_question(query):
        """Classify whether the question is general or domain-specific using Gemini."""
        response = model.generate_content([f"Classify this question as 'general' or 'domain-specific'. Give one word answer i.e general or domain-specific. General questions are greetings and questions involving general knowledge like the capital of France. General questions also involve politics, geography, history, economics, cosmology, information about famous personalities, etc. Question: {query}"], stream=True)
        response.resolve()
        return response.text.strip().lower()  # Assuming the response is either 'general' or 'domain-specific'

    # Main interaction loop in the Streamlit interface
    query = get_user_input()

    if query:
        # Add user input to conversation history
        conversation_history.append(f"You: {query}")

        # Step 1: Check if the question is about the system
        if classify_system_question(query):
            text = "I am an AI assistant capable of answering queries related to the National Building Code of Canada and general questions. I was developed by research group [SITE] at University of Alberta. How can I assist you further?"

        else:
            # Step 2: Classify the question as general or domain-specific
            question_type = classify_question(query)
            
            # If the question is general, use Gemini to directly answer it
            if question_type == "general":
                text = model.generate_content([f"Answer this general question: {query}. If it is greeting respond accordingly and if it is not greeting add a prefix saying that it is a general query."], stream=True)
                text.resolve()
                text = text.text

            else:
                # Step 3: Query the RAG model for domain-specific answers
                results = RAG.search(query, k=3)
                
                # Check if RAG found any results
                if not results:
                    text = model.generate_content([f"Answer this question: {query}"], stream=True)
                    text.resolve()
                    text = text.text
                    text = "It is a general query. ANSWER:" + text
                else:
                    # Process images from the results
                    image = process_image_from_results(results)
                    
                    # Generate the answer using the Gemini model if an image is found
                    if image:
                        text = generate_answer(query, image)
                        text = "It is a query from NBCC. ANSWER:" + text
                        
                        # Check if the answer is a fallback message (indicating no relevant answer)
                        if any(keyword in text.lower() for keyword in [
                            "does not provide", 
                            "cannot answer", 
                            "does not contain", 
                            "no relevant answer", 
                            "not found", 
                            "information unavailable", 
                            "not in the document", 
                            "unable to provide", 
                            "no data", 
                            "missing information", 
                            "no match", 
                            "provided text does not describe",
                            "are not explicitly listed",
                            "are not explicitly mentioned",
                            "no results", 
                            "not available", 
                            "query not found"
                        ]):
                            # Fallback to Gemini for answering
                            text = model.generate_content([f"Answer this general question in concise manner: {query}"], stream=True)
                            text.resolve()
                            text = text.text
                            text = "It is a general query. ANSWER: " + text

                    else:
                        text = model.generate_content([f"Answer this question: {query}"], stream=True)
                        text.resolve()
                        text = text.text
                        text = "It is a query from NBCC. ANSWER: " + text

        # Add the model's response to the conversation history
        conversation_history.append(f"Model: {text}")
        
        # Display the conversation history
        st.write("\n--- Conversation History ---")
        for message in conversation_history:
            st.write(message)

else:
    st.write("Please enter your Google API Key to continue.")
