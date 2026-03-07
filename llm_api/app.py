from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# In-memory session storage
# key = session_id (string UUID)
# value = list of message objects
sessions = {}

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable is not set!")
    print("Please ensure your .env file has OPENAI_API_KEY defined")
else:
    print("✓ OpenAI API key loaded successfully from .env file")

client = OpenAI(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, strip_whitespace=True)

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=1000, strip_whitespace=True)

class ChatResponse(BaseModel):
    response: str
    turn_count: int

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