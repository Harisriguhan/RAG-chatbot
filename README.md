# RAG Bot with Pinecone & Multiple Data Sources

A high-performance RAG chatbot with Pinecone vector database and multiple data source options.

## Setup

1. **Install Ollama**: https://ollama.ai
2. **Pull Llama3**: `ollama pull llama3`
3. **Start Ollama**: `ollama serve`
4. **Create Pinecone Account**: https://www.pinecone.io/
5. **Get API Keys**: Get your Pinecone API key and environment from the dashboard
6. **Install Dependencies**: `pip install -r requirements.txt`
7. **Configure Secrets**: 
   - For local development: Create `.streamlit/secrets.toml` with your Pinecone credentials
   - For production: Set `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` environment variables
8. **Run the app**: `streamlit run app.py`

## Data Sources Available

1. **Web URL**: Fetch content from any web page
2. **Wikipedia**: Retrieve Wikipedia articles on any topic
3. **YouTube**: Get transcripts from YouTube videos
4. **PDF File**: Upload and process PDF documents
5. **Text Input**: Paste custom text content
6. **Sample Data**: Load pre-prepared machine learning content

## Usage

1. **Add Content**: Use the sidebar to select a data source and add content to Pinecone
2. **Ask Questions**: Type questions in the chat interface
3. **Get Answers**: The bot will retrieve relevant information from Pinecone and generate responses using Llama3

## Features

- **Multiple Data Sources**: Web, Wikipedia, YouTube, PDF, text input, and sample data
- **Pinecone Vector Database**: High-speed, scalable vector storage and retrieval
- **Local LLM**: Uses Ollama & Llama3 (no API keys needed for generation)
- **Fast Semantic Search**: Pinecone provides millisecond response times
- **User-Friendly Interface**: Easy data source selection and management

## Configuration

### Local Development
Create `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your-api-key-here"
PINECONE_ENVIRONMENT = "your-environment-here"