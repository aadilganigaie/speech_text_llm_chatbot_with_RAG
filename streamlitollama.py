import streamlit as st
from langchain_postgres import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from sqlalchemy import create_engine, text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser
from langchain_core.documents.base import Blob
import os
from datetime import datetime
import plotly.express as px
import time
from io import BytesIO
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import sounddevice as sd
import tempfile
import queue
import requests
import threading
import base64 

# Add after existing imports
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add at the beginning of the file with other global variables
audio_semaphore = threading.Semaphore(1)

# Set page config
st.set_page_config(
    page_title="Document Search & Chat",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    /* Advanced container styling */
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Enhanced chat message styling */
    .stChatMessage {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
    }
    
    /* Custom message roles */
    [data-testid="StChatMessage"][data-role="human"] {
        border-left: 4px solid #007bff;
    }
    
    [data-testid="StChatMessage"][data-role="assistant"] {
        border-left: 4px solid #28a745;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem 1rem;
    }
    
    /* Animated buttons */
    .stButton>button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Custom metrics */
    .stMetric {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.05);
    }
    
    /* Custom expander */
    .stExpander {
        border: none;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* Chat input styling */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 0.8rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    /* Loading animation */
    .stSpinner {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: .5;
        }
    }
    
    .response-audio {
        margin: 10px 0;
        padding: 10px;
        border-radius: 10px;
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
    }
    
    .response-audio audio {
        width: 100%;
        border-radius: 10px;
        outline: none;
    }
    
    /* Hide default audio controls but keep functionality */
    .response-audio audio::-webkit-media-controls-panel {
        background: transparent;
    }
    
    /* Auto-play indicator */
    .response-audio::before {
        content: '🔊';
        position: absolute;
        right: 10px;
        top: 10px;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Fixed chat input container */
    .fixed-chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 100;
    }
    
    /* Add padding to main container to prevent content from being hidden behind fixed input */
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Custom container for input elements */
    .input-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize models
@st.cache_resource
def init_models():

    embeddings_model = OllamaEmbeddings(model = 'nomic-embed-text', base_url = 'http://sestrilevante.platform.myw.ai:11434')

    chat_model = ChatOllama(
        model='deepseek-r1:32b',
        base_url='http://sestrilevante.platform.myw.ai:11434',
        temperature=0.7
    )
    
    return embeddings_model, chat_model

@st.cache_resource
def init_STT_model():
    endpoint = "***"
    key = "83msI0RzecQT****"
    version = "2024-06-01"
    name = "whisper"
    parser = AzureOpenAIWhisperParser(api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name)
    return parser

# Add after init_STT_model()
@st.cache_resource
def init_TTS_model():
    """Initialize Azure TTS configuration"""
    return {
        "endpoint": "https://****,
        "api_key": "FD3zRAvrda5***",
        "api_version": "2024-05-01-preview"
    }

def text_to_speech(text, voice):
    """Convert text to speech using Azure TTS"""
    try:
        tts_config = init_TTS_model()
        
        headers = {
            "api-key": tts_config["api_key"],
            "Content-Type": "application/json"
        }
        
        # Updated payload without response_format
        payload = {
            "input": text,
            "voice": voice  # Using one of the allowed voices: nova, shimmer, echo, onyx, fable, alloy
        }
        
        logger.debug(f"Sending TTS request with payload: {payload}")
        
        response = requests.post(
            f"{tts_config['endpoint']}?api-version={tts_config['api_version']}", 
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"TTS Error: {response.status_code} - {response.text}")
            # Add more detailed error logging
            if response.text:
                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    logger.error(f"Raw error response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"TTS processing error: {str(e)}", exc_info=True)
        return None

parser = init_STT_model()

def process_audio_input(audio_bytes):
    """Process audio input and convert to text using Azure Whisper"""
    temp_file = None
    try:
        # Create a temporary file with a unique name
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        
        # Close the file handle immediately after writing
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Create blob from the temp file
        audio_blob = Blob.from_path(temp_path)
        
        # Get the parser and convert speech to text
        parser = init_STT_model()
        docs = parser.parse(audio_blob)
        
        if docs and len(docs) > 0:
            return docs[0].page_content
        return None
            
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}", exc_info=True)
        st.error(f"Error processing audio: {str(e)}")
        return None
        
    finally:
        # Ensure file cleanup happens in finally block
        if temp_file:
            try:
                temp_file.close()
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {str(e)}")

def record_audio():
    """Record audio from computer microphone"""
    # Audio parameters
    sample_rate = 44100
    duration = 10  # max recording duration in seconds
    channels = 1
    
    # Create a queue for audio data
    audio_queue = queue.Queue()
    recording = threading.Event()
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())
    
    # Create columns for recording controls
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🎤 Record"):
            recording.set()
            audio_data = []
            
            try:
                with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
                    with st.spinner("🔴 Recording... (Click Stop when done)"):
                        while recording.is_set():
                            audio_data.append(audio_queue.get())
                    
                if audio_data:
                    # Combine all audio chunks
                    audio_data = np.concatenate(audio_data)
                    
                    # Save to temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        sf.write(temp_audio.name, audio_data, sample_rate)
                        return temp_audio.name
                        
            except Exception as e:
                st.error(f"Recording error: {str(e)}")
                return None
                
    with col2:
        if st.button("⏹️ Stop"):
            recording.clear()

# Database connection
@st.cache_resource
def init_db():
    connection_string = "postgresql://***:****@localhost:5432/****"
    return create_engine(connection_string)

def stream_data(placeholder, text):
    try:
        sentences = text.split('. ')
        message = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            message += sentence.strip() + ". "
            placeholder.markdown(message)
            time.sleep(0.1)
        
        return message
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")
        return text  # Fallback to displaying full text

# Search function
def search_documents(query, embeddings_model, engine, top_k=5):
    query_embedding = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
    
    with engine.connect() as conn:
        # Fixed SQL query with correct column name casing
        results = conn.execute(text("""
            SELECT embedding, "pageContent"
            FROM langchain_pg_embedding
        """)).fetchall()
        
        embeddings = []
        contents = []
        for row in results:
            embedding_str = row[0]
            embedding_values = embedding_str.strip('[]').split(',')
            embedding_array = np.array([float(x.strip()) for x in embedding_values])
            embeddings.append(embedding_array)
            contents.append(row[1])
        
        stored_vectors = np.stack(embeddings)
        similarities = cosine_similarity(query_embedding, stored_vectors)
        most_similar_idx = np.argsort(similarities[0])[::-1][:top_k]
        
        return [(similarities[0][idx], contents[idx]) for idx in most_similar_idx]

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add near the top where other session states are initialized
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

def search_and_get_context(query, embeddings_model, engine, top_k=3):
    results = search_documents(query, embeddings_model, engine, top_k)
    # Show similar documents in expander
    with st.expander("📑 Similar Documents", expanded=False):
        for i, (score, content) in enumerate(results, 1):
            st.markdown(f"**Document {i}** (Similarity: {score:.4f})")
            st.markdown(f"```\n{content[:300]}...\n```")
            st.markdown("---")
    
    # Return contexts for chat
    contexts = [content for _, content in results]
    return "\n\n".join(contexts)

# Add to your main chat display code
def display_message_with_animation(message, index):
    with st.chat_message(message["role"]):
        st.markdown(
            f"""
            <div class="message-animation" 
                 style="animation: slideIn 0.3s ease-out {index * 0.1}s both;">
                {message["content"]}
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.caption(f"Time: {message.get('timestamp', 'N/A')}")

def stream_speech(text_chunks, voice):
    """Stream text chunks to speech in real-time with sequential playback"""
    for chunk in text_chunks:
        if chunk.strip():  # Only process non-empty chunks
            try:
                audio_response = text_to_speech(chunk, voice)
                if audio_response:
                    with audio_semaphore:  # Ensure sequential playback
                        with st.container():
                            # Generate unique ID for each audio element
                            audio_id = f"audio_{hash(chunk)}"
                            st.markdown(f"""
                                <div class="response-audio">
                                    <audio id="{audio_id}" onended="document.getElementById('{audio_id}').nextElementSibling?.play()" autoplay>
                                        <source src="data:audio/wav;base64,{base64.b64encode(audio_response).decode()}" type="audio/wav">
                                    </audio>
                                </div>
                            """, unsafe_allow_html=True)
                            # Wait for current audio to finish before proceeding
                            time.sleep(len(chunk.split()) * 0.3)  # Approximate timing based on word count
            except Exception as e:
                logger.error(f"Streaming TTS error: {str(e)}")
                continue

def stream_response(response_iterator, input_type, voice=None):
    """Stream model response with simultaneous TTS conversion"""
    full_response = ""
    buffer = ""
    
    # Create placeholder for streaming
    message_placeholder = st.empty()
    
    for chunk in response_iterator:
        if chunk:
            # Extract text content from AIMessageChunk
            chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_response += chunk_text
            buffer += chunk_text
            
            # For text mode, show streaming text
            if input_type == "Text":
                message_placeholder.markdown(full_response)
            else:
                # For audio mode, convert complete sentences to speech
                if '.' in buffer or '!' in buffer or '?':
                    sentences = buffer.split('.')
                    complete_sentences = sentences[:-1]
                    buffer = sentences[-1]
                    
                    if complete_sentences:
                        stream_speech(complete_sentences, voice)
    
    # Process any remaining text in buffer
    if buffer and input_type == "Audio":
        stream_speech([buffer], voice)
    
    return full_response

# Main app
def main():
    # Initialize models and database first
    embeddings_model, chat_model = init_models()
    engine = init_db()

    # Page header with custom layout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💬 AI Document Assistant")
    with col2:
        st.image("https://img.icons8.com/clouds/100/000000/chat.png", width=100)
    
    # Add session state for storing model temperature
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    # Sidebar enhancements
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Quick Actions Section
        st.subheader("Quick Actions")
        if st.button("📝 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()  # Visual separator
        
        # Search settings
        st.subheader("Search Configuration")
        top_k = st.slider("Number of similar documents", 1, 10, 3)
        min_similarity = st.slider("Minimum similarity score", 0.0, 1.0, 0.5)
        
        # Chat settings
        st.subheader("Chat Configuration")
        st.session_state.temperature = st.slider("Response creativity", 0.0, 1.0, st.session_state.temperature)
        
        st.divider()  # Visual separator
        
        # Analytics section in sidebar
        if st.session_state.messages:
            st.subheader("Chat Analytics")
            num_messages = len(st.session_state.messages)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", num_messages)
            with col2:
                human_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "human")
                st.metric("Questions", human_msgs)
        
        # Move clear chat to bottom of sidebar
        st.divider()
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            confirm = st.button("⚠️ Confirm Clear?", type="primary")
            if confirm:
                st.session_state.messages = []
                st.rerun()

        # TTS settings
        st.subheader("Voice Settings")
        voice_option = st.selectbox(
            "Choose voice",
            options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0,
            help="Select the voice for audio responses"
        )

        # Store voice selection in session state
        if "tts_voice" not in st.session_state:
            st.session_state.tts_voice = voice_option
        else:
            st.session_state.tts_voice = voice_option

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💭 Chat", "📊 Analytics", "ℹ️ Help"])
    
    with tab1:
        # Chat container with fixed height for scrolling
        chat_container = st.container()
        st.markdown("""
            <style>
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                height: calc(100vh - 180px);
                overflow-y: auto;
                margin-bottom: 80px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        with chat_container:
            # Display chat history with timestamps
            for idx, message in enumerate(st.session_state.messages):
                display_message_with_animation(message, idx)
        
        # Fixed chat input at bottom
        st.markdown("""
            <div class="fixed-chat-input">
                <div class="input-container">
        """, unsafe_allow_html=True)
        
        # Initialize prompt variable
        prompt = None
        
        # Audio/Text input toggle
        input_type = st.radio(
            "Choose input method:",
            ["Text", "Audio"],
            horizontal=True,
            key="input_type",
            label_visibility="collapsed"
        )
        
        if input_type == "Text":
            prompt = st.chat_input("Ask me anything about the documents...", key="chat_input")
        else:
            st.markdown("""
                <style>
                .audio-recorder { 
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 10px;
                    background: linear-gradient(145deg, #f8f9fa, #ffffff);
                    box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
                }
                </style>
            """, unsafe_allow_html=True)
            
            audio_bytes = audio_recorder(
                pause_threshold=2.0,
                sample_rate=44100,
                text="🎤 Click to start recording",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if audio_bytes:
                try:
                    # Display audio player
                    st.audio(audio_bytes, format="audio/wav")
                    
                    with st.spinner("🎧 Processing audio..."):
                        # Process the audio
                        transcribed_text = process_audio_input(audio_bytes)
                        
                        if transcribed_text:
                            st.success("Audio transcribed successfully!")
                            st.info(f"Transcribed text: {transcribed_text}")
                            prompt = transcribed_text
                        else:
                            st.error("Could not transcribe audio. Please try again.")
                            prompt = None
                            
                except Exception as e:
                    logger.error(f"Audio recording error: {str(e)}", exc_info=True)
                    st.error("Error processing audio. Please try again.")
                    prompt = None

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Process prompt if it exists
        if prompt:
            # Rest of your existing code for handling the prompt
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "human", 
                "content": prompt,
                "timestamp": timestamp
            })
            
            with st.chat_message("human"):
                st.markdown(prompt)
                st.caption(f"Time: {timestamp}")
            
            # Search and display context
            with st.spinner("🔍 Searching documents..."):
                context = search_and_get_context(prompt, embeddings_model, engine, top_k)
                
            # Enhanced response generation
            with st.spinner("🤔 Thinking..."):
                messages = [
                    SystemMessage(content=f"""You are a helpful and friendly assistant. 
                    Use the following context to answer questions thoroughly yet concisely. 
                    If you're not sure about something, be honest about it.
                    
                    Context: {context}"""),
                ]
                
                # Add conversation history
                for message in st.session_state.messages[:-1]:
                    if message["role"] == "human":
                        messages.append(HumanMessage(content=message["content"]))
                    else:
                        messages.append(AIMessage(content=message["content"]))
                
                # Add current prompt
                messages.append(HumanMessage(content=prompt))
                
                # Get response from Ollama
                response = chat_model.invoke(messages)
            
            # Replace the existing assistant response section
            with st.chat_message("assistant"):
                if input_type == "Text":
                    full_response = stream_response(
                        chat_model.stream(messages),
                        input_type="Text"
                    )
                else:
                    st.markdown("""
                        <style>
                        .response-audio {
                            margin: 10px 0;
                            padding: 10px;
                            border-radius: 10px;
                            background: linear-gradient(145deg, #f8f9fa, #ffffff);
                            box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
                        }
                        audio {
                            width: 100%;
                            border-radius: 10px;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    full_response = stream_response(
                        chat_model.stream(messages),
                        input_type="Audio",
                        voice=st.session_state.tts_voice
                    )
                
                # Show timestamp
                current_time = datetime.now().strftime("%H:%M:%S")
                st.caption(f"Time: {current_time}")
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "timestamp": current_time
            })

    with tab2:
        # Analytics dashboard
        st.subheader("Chat Analytics Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            # Message count over time
            if st.session_state.messages:
                timestamps = [msg.get('timestamp') for msg in st.session_state.messages]
                msg_counts = range(1, len(timestamps) + 1)
                fig = px.line(x=timestamps, y=msg_counts, 
                            title="Conversation Progress",
                            labels={"x": "Time", "y": "Messages"})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time analysis
            if len(st.session_state.messages) > 1:
                response_times = []
                for i in range(1, len(st.session_state.messages), 2):
                    try:
                        t1 = datetime.strptime(st.session_state.messages[i-1]['timestamp'], "%H:%M:%S")
                        t2 = datetime.strptime(st.session_state.messages[i]['timestamp'], "%H:%M:%S")
                        response_times.append((t2-t1).seconds)
                    except:
                        continue
                if response_times:
                    fig = px.box(y=response_times, 
                               title="Response Time Distribution (seconds)")
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Help section
        st.markdown("""
        ### 🤖 How to Use This ChatBot
        
        1. **Ask Questions**: Type your question in the chat input below
        2. **View Similar Documents**: Check the expander to see relevant documents
        3. **Adjust Settings**: Use the sidebar to customize the experience
        
        ### 🎯 Features
        
        - **Document Search**: Advanced similarity search in your documents
        - **Context Awareness**: Bot remembers conversation history
        - **Analytics**: Track your conversation statistics
        - **Customizable**: Adjust various parameters in settings
        
        ### 💡 Tips
        
        - Be specific in your questions
        - Check similar documents for context
        - Adjust temperature for different response styles
        """)

if __name__ == "__main__":
    main()
