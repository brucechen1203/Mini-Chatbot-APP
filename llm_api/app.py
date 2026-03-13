from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints
import uvicorn
import os
import uuid
from typing import Annotated, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# In-memory session storage
# key = session_id (string UUID)
# value = list of message objects
sessions = {}

# In-memory chunk storage for RAG
# each item: {chunk_id, doc_id, text, embedding}
chunks = []

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable is not set!")
    print("Please ensure your .env file has OPENAI_API_KEY defined")
else:
    print("✓ OpenAI API key loaded successfully from .env file")

client = OpenAI(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]

class ChatResponse(BaseModel):
    response: str
    turn_count: int

class IngestRequest(BaseModel):
    doc_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    Uses character-based slicing for a simple baseline implementation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    step = chunk_size - overlap
    result = []

    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()
        if chunk:
            result.append(chunk)
        start += step

    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mini-Chatbot API!"}

@app.post("/session")
def create_session():
    """
    Create a new chat session with unique ID.
    Initializes conversation history with system prompt.
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize conversation history with system prompt
    sessions[session_id] = [
        {"role": "system", "content": "You are a helpful CS teaching assistant. Give concise explanations."}
    ]
    
    print(f"Created new session: {session_id}")
    
    return {"session_id": session_id}

@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    """
    Ingest a document for RAG.
    - Split text into overlapping fixed-size chunks
    - Generate embeddings for each chunk
    - Store chunk text, metadata, and vectors in memory
    """
    try:
        text_chunks = split_text_into_chunks(request.text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Document text is empty after preprocessing")

        # Remove existing chunks for this doc_id to avoid duplicates on re-ingestion.
        global chunks
        chunks = [chunk for chunk in chunks if chunk["doc_id"] != request.doc_id]

        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text_chunks
        )

        if len(embedding_response.data) != len(text_chunks):
            raise HTTPException(status_code=500, detail="Embedding count does not match chunk count")

        for index, (chunk_text, embedding_item) in enumerate(zip(text_chunks, embedding_response.data), start=1):
            chunk_id = f"{request.doc_id}#{index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": request.doc_id,
                    "text": chunk_text,
                    "embedding": embedding_item.embedding,
                }
            )

        print(f"Ingested doc_id={request.doc_id} with {len(text_chunks)} chunks")

        return {
            "doc_id": request.doc_id,
            "chunks_added": len(text_chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while ingesting the document")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Handle multi-turn conversation.
    Maintains conversation history and sends full context to LLM.
    """
    # Validate session exists
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Retrieve conversation history
    conversation_history = sessions[request.session_id]
    
    # Append user message to history
    conversation_history.append({
        "role": "user",
        "content": request.message
    })
    
    try:
        # Send full conversation history to LLM
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )
        
        # Extract assistant response text from SDK result
        assistant_message = response.choices[0].message.content
        if not assistant_message:
            raise HTTPException(status_code=500, detail="Model returned an empty response")
        
        # Append assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Count completed user turns.
        turn_count = len([msg for msg in conversation_history if msg["role"] == "user"])
        
        print(f"Session {request.session_id} - Turn {turn_count}")
        print(f"User: {request.message}")
        print(f"Assistant: {assistant_message}")
        print("\n" + "="*50 + "\n")
        
        return {
            "response": assistant_message,
            "turn_count": turn_count
        }
    
    except Exception as e:
        # Log the error and return 500
        print(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

@app.post("/test")
def test_prompt(request: PromptRequest):
    print(f"Received request: {request}")
    print(f"Prompt: {request.prompt}")
    return {"received_prompt": request.prompt, "message": "JSON parsed successfully!"}

@app.post("/hello")
def hello(request: PromptRequest):
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": request.prompt}
            ]
        )
        
        # Extract the generated text from the model
        model_output = response.choices[0].message.content
        
        # Print extracted text for verification
        print(f"Original Prompt: {request.prompt}")
        print(f"Model Output: {model_output}")
        print("\n" + "="*50 + "\n")
        
        return {
            "original_prompt": request.prompt,
            "model_output": model_output
        }
    except Exception as e:
        # Log the error internally but don't expose details
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)