import os
import pickle
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# Set HuggingFace API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_TOKEN"

# Function to process and query documents from multiple URLs
def process_and_query(urls):
    # List to hold all the documents
    all_docs = []

    # Iterate over each URL and load the data
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=2000,
                                              chunk_overlap=500)
        docs = text_splitter.split_documents(data)

        # Add the documents to the all_docs list
        all_docs.extend(docs)

    # Create Hugging Face embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create a FAISS vector database from all the documents
    vectordb = FAISS.from_documents(documents=all_docs, embedding=huggingface_embeddings)

    return vectordb

# Streamlit interface
def main():
    # Set page title and icon
    st.set_page_config(page_title="Conversational Chatbot with for URLs", page_icon=":speech_balloon:")
    st.title("Conversational Chatbot with for URLs")

    # Get URLs from the user as a comma-separated input
    user_urls = st.text_area("Enter URLs (separated by commas):")

    # Add a submit button for URL processing
    if st.button("Submit URLs"):
        if user_urls:
            urls = [url.strip() for url in user_urls.split(',')]

            # Process URLs only once
            with st.spinner("Processing the URLs..."):
                vectordb = process_and_query(urls)
            st.success("URLs processed successfully! Ask your questions below.")

            # Store vector database in session state to avoid re-processing
            st.session_state.vectordb = vectordb

            # Initialize conversation buffer memory for chat history
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Create a retriever from the FAISS vector database
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})

            # Use a Hugging Face model
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

            # Create a ConversationalRetrievalChain from the model, retriever, and memory
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Check if the QA chain is created before proceeding to the chat
    if "qa_chain" in st.session_state:
        # Initialize or load chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display the chat messages
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        # User input for questions
        user_query = st.chat_input("Type a message")
        if user_query and user_query.strip() != "":
            # Add human message to the chat history
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            # Display human message
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Get and display the AI response without explanation part
            with st.chat_message("AI"):
                response = st.session_state.qa_chain({"question": user_query})
                st.markdown(response['answer'])  # Removed explanation part

            # Add AI message to the chat history
            st.session_state.chat_history.append(AIMessage(content=response['answer']))

if __name__ == '__main__':
    main()
