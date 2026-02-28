from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable is not set!")
    print("Please ensure your .env file has OPENAI_API_KEY defined")
else:
    print("✓ OpenAI API key loaded successfully from .env file")

client = OpenAI(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, strip_whitespace=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mini-Chatbot API!"}

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