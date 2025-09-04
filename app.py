"""
RAG Bot with Ollama, Llama3, Pinecone & Multiple Data Sources
A high-performance RAG chatbot with various data source options.
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import uuid
from datetime import datetime
import tempfile
import nltk
import ollama
nltk.download('punkt')

# Try to import Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Try to import data source libraries
try:
    #from llama_index import download_loader
    from llama_index.readers.web import SimpleWebPageReader
    from llama_index.readers.wikipedia import WikipediaReader
    from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
    from llama_index.readers.file import PDFReader
    DATA_SOURCES_AVAILABLE = True
except ImportError:
    DATA_SOURCES_AVAILABLE = False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pinecone_initialized" not in st.session_state:
    st.session_state.pinecone_initialized = False
if "knowledge_base_stats" not in st.session_state:
    st.session_state.knowledge_base_stats = {"chunks_count": 0, "last_updated": None}

# App configuration
st.set_page_config(page_title="RAG Bot with Pinecone", page_icon="🤖", layout="wide")

# Initialize Pinecone
def init_pinecone():
    if not PINECONE_AVAILABLE:
        st.sidebar.error("Pinecone not available. Install with: pip install pinecone-client")
        return False
    
    try:
        # Try to get from secrets first, then environment variables
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY"))
        
        if not pinecone_api_key:
            st.sidebar.error("Pinecone API key not set. Please configure it.")
            return False
        
        # Initialize Pinecone client with new SDK
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist
        index_name = "rag-bot-index"
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            # Create new index
            pc.create_index(
                name=index_name,
                dimension=384,  # Match all-MiniLM-L6-v2 embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            import time
            time.sleep(1)
        
        # Connect to index
        st.session_state.pinecone_index = pc.Index(index_name)
        st.session_state.pinecone_initialized = True
        return True
    except Exception as e:
        st.sidebar.error(f"Error initializing Pinecone: {str(e)}")
        return False


# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Process text into chunks
def chunk_text(text, chunk_size=500):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Generate embeddings
def get_embedding(text, model):
    return model.encode([text])[0]

# Add documents to Pinecone
def add_to_pinecone(text, model, source_name="Custom"):
    if not st.session_state.pinecone_initialized:
        if not init_pinecone():
            return False
    
    chunks = chunk_text(text)
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, model)
        vector_id = f"{source_name}_{i}_{uuid.uuid4().hex[:8]}"
        vectors.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": {"text": chunk, "source": source_name}
        })
    
    # Upsert vectors to Pinecone in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        st.session_state.pinecone_index.upsert(vectors=batch)
    
    # Update stats
    st.session_state.knowledge_base_stats["chunks_count"] += len(chunks)
    st.session_state.knowledge_base_stats["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    st.sidebar.success(f"Added {len(chunks)} chunks from {source_name} to Pinecone")
    return True

# Retrieve relevant chunks from Pinecone
def retrieve_chunks_pinecone(query, model, top_k=3):
    if not st.session_state.pinecone_initialized:
        if not init_pinecone():
            return ["Pinecone not initialized. Please check configuration."]
    
    query_embedding = get_embedding(query, model).tolist()
    
    try:
        results = st.session_state.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if results and 'matches' in results:
            return [match['metadata']['text'] for match in results['matches']]
        return ["No results found in knowledge base."]
    except Exception as e:
        return [f"Error querying Pinecone: {str(e)}"]

def generate_response(query, context_chunks):
    # Filter out error messages from context chunks
    valid_chunks = [chunk for chunk in context_chunks if not chunk.startswith("Error") and not chunk.startswith("Pinecone")]
    
    if not valid_chunks:
        return "I don't have enough information to answer this question. Please add some content to the knowledge base first."
    
    context = "\n".join([f"- {chunk}" for chunk in valid_chunks])
    prompt = f"""Based on this context:
{context}

Answer this question: {query}

If the context doesn't contain the answer, say you don't know."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


# Fetch content from web URL
def fetch_web_content(url):
    if not DATA_SOURCES_AVAILABLE:
        return "Data source libraries not available. Install with: pip install llama-index llama-index-readers-web"
    
    try:
        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=[url])
        return "\n".join([doc.text for doc in documents])
    except Exception as e:
        return f"Error fetching web content: {str(e)}"

# Fetch content from Wikipedia
def fetch_wikipedia_content(topic):
    if not DATA_SOURCES_AVAILABLE:
        return "Data source libraries not available. Install with: pip install llama-index llama-index-readers-wikipedia"
    
    try:
        loader = WikipediaReader()
        documents = loader.load_data(pages=[topic])
        return "\n".join([doc.text for doc in documents])
    except Exception as e:
        return f"Error fetching Wikipedia content: {str(e)}"# Enhanced Wikipedia content fetching with better error handling

def fetch_wikipedia_content(topic):
    if not DATA_SOURCES_AVAILABLE:
        return "Data source libraries not available. Install with: pip install llama-index llama-index-readers-wikipedia"
    
    try:
        # Clean the topic input
        cleaned_topic = topic.strip()
        
        # First try the direct approach with LlamaIndex's WikipediaReader
        try:
            loader = WikipediaReader()
            documents = loader.load_data(pages=[cleaned_topic])
            
            if documents and documents[0].text.strip():
                return "\n".join([doc.text for doc in documents])
        except Exception as e:
            st.warning(f"Direct Wikipedia access failed: {str(e)}")
            # Fall back to alternative method
        
        # Alternative approach using wikipedia-api
        try:
            import wikipediaapi
            wiki_wiki = wikipediaapi.Wikipedia(
                user_agent='RAG-Bot/1.0 (your-email@example.com)'
            )
            
            page = wiki_wiki.page(cleaned_topic)
            if page.exists():
                return page.text
            else:
                return f"Wikipedia page '{cleaned_topic}' does not exist. Try a different topic."
                
        except ImportError:
            # Fallback to requests-based approach
            try:
                import requests
                # Use Wikipedia API directly
                url = f"https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": cleaned_topic,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                pages = data["query"]["pages"]
                page_id = list(pages.keys())[0]
                
                if page_id != "-1":  # -1 means page doesn't exist
                    return pages[page_id]["extract"]
                else:
                    return f"Wikipedia page '{cleaned_topic}' does not exist. Try a different topic."
                    
            except Exception as e:
                return f"Error fetching Wikipedia content: {str(e)}"
                
    except Exception as e:
        return f"Error fetching Wikipedia content: {str(e)}"

# Fetch YouTube transcript
def fetch_youtube_transcript(video_url):
    if not DATA_SOURCES_AVAILABLE:
        return "Data source libraries not available. Install with: pip install llama-index llama-index-readers-youtube-transcript"
    
    try:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=[video_url])
        return "\n".join([doc.text for doc in documents])
    except Exception as e:
        return f"Error fetching YouTube transcript: {str(e)}"

# Process PDF file
def process_pdf_file(uploaded_file):
    if not DATA_SOURCES_AVAILABLE:
        return "Data source libraries not available. Install with: pip install llama-index"
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PDFReader()
        documents = loader.load_data(file=tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return "\n".join([doc.text for doc in documents])
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Sample data function
def load_sample_data():
    sample_text = """
    Machine Learning Basics
    
    Machine learning is a subset of artificial intelligence that focuses on 
    building systems that learn from data. There are three main types:
    
    1. Supervised Learning: Learning from labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data  
    3. Reinforcement Learning: Learning through rewards and penalties
    
    Common algorithms include decision trees, neural networks, and SVMs.
    Deep learning uses multi-layer neural networks for complex tasks like
    image recognition and natural language processing.
    
    The machine learning workflow typically involves data collection, 
    preprocessing, model training, evaluation, and deployment. Quality data 
    is essential for building effective models.
    
    Applications of machine learning include:
    - Healthcare: Disease prediction and medical image analysis
    - Finance: Fraud detection and algorithmic trading
    - Retail: Recommendation systems and inventory management
    - Transportation: Self-driving cars and route optimization
    """
    
    model = load_embedder()
    if add_to_pinecone(sample_text, model, "Sample Data"):
        st.sidebar.success("Sample data loaded to Pinecone!")

# Sidebar for adding content
with st.sidebar:
    st.title("🤖 RAG Bot with Pinecone")
    
    # Initialize Pinecone
    if not st.session_state.pinecone_initialized:
        init_pinecone()
    
    if not PINECONE_AVAILABLE:
        st.warning("Pinecone client not installed.")
        st.code("pip install pinecone-client")
    
    if not DATA_SOURCES_AVAILABLE:
        st.warning("Data source libraries not installed.")
        st.code("pip install llama-index llama-index-readers-web llama-index-readers-wikipedia llama-index-readers-youtube-transcript")
    
    st.subheader("Add Data to Knowledge Base")
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Web URL", "Wikipedia", "YouTube", "PDF File", "Text Input", "Sample Data"],
        horizontal=True
    )
    
    model = load_embedder()
    
    if data_source == "Web URL":
        st.write("**Fetch content from a web page**")
        url = st.text_input("Enter web URL:", placeholder="https://example.com")
        if st.button("Fetch Web Content") and url:
            with st.spinner("Fetching content..."):
                content = fetch_web_content(url)
                if not content.startswith("Error"):
                    if add_to_pinecone(content, model, f"Web: {url}"):
                        st.success("Content added to knowledge base!")
                else:
                    st.error(content)
    
    elif data_source == "Wikipedia":
        st.write("**Fetch content from Wikipedia**")
        topic = st.text_input("Enter Wikipedia topic:", placeholder="Machine learning")
        if st.button("Fetch Wikipedia Article") and topic:
            with st.spinner("Fetching article..."):
                content = fetch_wikipedia_content(topic)
                if not content.startswith("Error"):
                    if add_to_pinecone(content, model, f"Wikipedia: {topic}"):
                        st.success("Content added to knowledge base!")
                else:
                    st.error(content)
    
    elif data_source == "YouTube":
        st.write("**Fetch transcript from YouTube video**")
        video_url = st.text_input("Enter YouTube video URL:", placeholder="https://youtube.com/watch?v=...")
        if st.button("Fetch Transcript") and video_url:
            with st.spinner("Fetching transcript..."):
                content = fetch_youtube_transcript(video_url)
                if not content.startswith("Error"):
                    if add_to_pinecone(content, model, f"YouTube: {video_url}"):
                        st.success("Content added to knowledge base!")
                else:
                    st.error(content)
    
    elif data_source == "PDF File":
        st.write("**Upload and process a PDF file**")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                content = process_pdf_file(uploaded_file)
                if not content.startswith("Error"):
                    if add_to_pinecone(content, model, f"PDF: {uploaded_file.name}"):
                        st.success("Content added to knowledge base!")
                else:
                    st.error(content)
    
    elif data_source == "Text Input":
        st.write("**Add custom text content**")
        user_text = st.text_area("Paste text here:", height=150)
        if st.button("Add to Knowledge Base") and user_text:
            if add_to_pinecone(user_text, model, "Custom Text"):
                st.success("Content added to knowledge base!")
    
    else:  # Sample Data
        st.write("**Load sample machine learning content**")
        if st.button("Load Sample Data"):
            load_sample_data()
    
    # Knowledge base info
    st.divider()
    st.subheader("Knowledge Base Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", st.session_state.knowledge_base_stats["chunks_count"])
    with col2:
        if st.session_state.knowledge_base_stats['last_updated']:
            st.write("Last updated:")
            st.write(st.session_state.knowledge_base_stats['last_updated'])
    
    if st.button("Clear Knowledge Base", type="secondary"):
        if st.session_state.pinecone_initialized:
            # Note: In production, you might want to delete and recreate the index
            st.session_state.knowledge_base_stats = {"chunks_count": 0, "last_updated": None}
            st.success("Knowledge base stats cleared! (Pinecone data persists)")
        else:
            st.error("Pinecone not initialized")

# Main chat interface
st.header("💬 RAG Chatbot with Pinecone")
st.caption("Ask questions about content from various sources with fast Pinecone retrieval")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            model = load_embedder()
            relevant_chunks = retrieve_chunks_pinecone(prompt, model)
            
        with st.spinner("💭 Generating response..."):
            response = generate_response(prompt, relevant_chunks)
            
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show context sources
            with st.expander("📚 View sources used", expanded=False):
                if relevant_chunks and not any(
                    chunk.startswith(("Error", "Pinecone", "No results")) 
                    for chunk in relevant_chunks
                ):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Source {i+1}:**")
                        st.write(chunk[:250] + "..." if len(chunk) > 250 else chunk)
                        st.divider()
                else:
                    st.info("No sources found or error retrieving sources.")