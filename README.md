# Web Scrapping Chatbot - Conversational Chatbot with Web Scraping Using LLM and FAISS

## Overview

This project leverages **LangChain**, **FAISS**, and **Hugging Face** models to build a conversational chatbot that can scrape data from websites, process it, and then answer questions based on the scraped content. Users can provide multiple URLs, and the chatbot will allow them to ask queries using natural language, retrieving relevant information from the provided web pages.

The chatbot uses **sentence embeddings** from Hugging Face to create a searchable vector database of the scraped content and employs **Mistral 7B LLM** for generating responses. It also incorporates **memory** to maintain context in the conversation, making interactions more seamless and coherent.

## Features

- **Multi-URL Web Scraping**: Scrapes text content from multiple web pages and processes it for querying.
- **Conversational Interface**: Users can interact with the chatbot, asking questions in natural language about the scraped content.
- **Embeddings and Vector Search**: Uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` for creating embeddings and FAISS for storing and searching through the content.
- **LLM Integration**: The chatbot is powered by the **Mistral 7B LLM**, providing intelligent responses based on the content retrieved.
- **Chat History and Context**: The conversation retains context, allowing follow-up questions that reference prior discussion.
- **Efficient Chunking**: Automatically splits large documents into manageable chunks to improve embedding and retrieval efficiency.

## Technologies Used

- **LangChain**: A framework for building language model-based applications, especially for chaining multiple operations like scraping, embedding, and retrieval.
- **FAISS**: A fast library for similarity search and clustering of dense vectors.
- **Hugging Face Transformers**: For language model-based reasoning and conversation.
- **Streamlit**: A framework for building data-driven applications with interactive interfaces.
- **BeautifulSoup4**: A web scraping library to extract and parse website data.

## Installation

### Prerequisites

- Python 3.8+
- Hugging Face account (for using the API token)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/VRAJ-07/Web-Scrapping-Chatbot.git
    cd Web-Scrapping-Chatbot
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Hugging Face API key. Replace the `YOUR_HUGGINGFACE_API_TOKEN` placeholder in the code or in the environment with your Hugging Face token:

    ```bash
    export HUGGINGFACEHUB_API_TOKEN=your_api_token_here
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```

### Example URLs

You can provide URLs for scraping such as:

- Articles or blog posts
- Documentation pages
- Any webpage containing text data that you want to query.

## Usage

1. **Launch the App**: Once the app is running, you'll see a text area to input URLs and a chat interface below it.
   
2. **Input URLs**: Enter one or more URLs (separated by commas) into the text area and click the **Submit URLs** button to start the scraping and processing.

3. **Ask Questions**: After the URLs are processed, you can ask questions related to the content of the web pages in the chat interface. The chatbot will respond based on the scraped information.

4. **Conversational Context**: You can continue the conversation by asking follow-up questions, and the chatbot will remember the context of the previous interactions.


## Example 
https://github.com/user-attachments/assets/222f4d46-f1c6-41aa-ad9d-c7b52df7e486

## Code Breakdown

### Main Components

- **Web Scraping**: The `WebBaseLoader` class is used to fetch the content from the provided URLs. This content is split into manageable chunks using a text splitter (`CharacterTextSplitter`), ensuring that large documents are efficiently processed.
  
- **Embeddings**: Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` is used to embed the document chunks, transforming them into vectors that can be stored and searched.

- **Vector Store**: **FAISS** is used as the vector database. It stores the embeddings and allows for efficient similarity search, enabling the chatbot to retrieve relevant chunks based on the user’s query.

- **Conversation Chain**: A **ConversationalRetrievalChain** is created using the **Mistral 7B LLM** from Hugging Face. This chain retrieves relevant document chunks, processes the user's question, and generates a coherent response while maintaining context across the conversation.

- **Chat Interface**: The conversational UI is built with Streamlit, where users can input URLs, ask questions, and view both their messages and the AI's responses in a clear and interactive format.

### Key Functions

- **`process_and_query(urls)`**: Scrapes content from the given URLs, splits the text into chunks, and embeds the chunks into vectors using Hugging Face embeddings.
  
- **`main()`**: The Streamlit interface, which handles user input for URLs and chat messages, and orchestrates the chatbot interactions.

## Future Improvements

- **Multi-modal Support**: Extend the chatbot to handle not just text but also images or multimedia content scraped from websites.
- **Pagination and Infinite Scroll**: Add support for long documents by implementing a pagination system in the chat interface.
- **Better Query Understanding**: Improve the chatbot’s ability to handle complex queries involving comparisons between multiple URLs.
- **Improved Chunking**: Implement advanced techniques for chunking documents, like semantic-based splitting, to improve response accuracy.
