from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints
import uvicorn
import os
import uuid
import math
import ast
import operator
import json
import re
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

class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class Citation(BaseModel):
    chunk_id: str
    score: float

class QARequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    k: int = Field(default=4, ge=1)

class QAResponse(BaseModel):
    answer: str
    citations: List[Citation]
    turn_count: int

class AgentRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    k: int = Field(default=4, ge=1)

class AgentStep(BaseModel):
    action: str
    input: str
    output: str

class AgentResponse(BaseModel):
    answer: str
    steps: List[AgentStep]

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

def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Compute cosine similarity manually (no external numeric libraries).
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def retrieve_top_chunks(query: str, k: int) -> List[dict]:
    """
    Retrieve top-k chunks by cosine similarity.
    """
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_embedding = query_embedding_response.data[0].embedding

    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
            }
        )

    ranked_results = sorted(scored_chunks, key=lambda item: item["score"], reverse=True)
    return ranked_results[:k]

ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _eval_arithmetic_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BIN_OPS:
        left_value = _eval_arithmetic_ast(node.left)
        right_value = _eval_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Div) and right_value == 0:
            raise ValueError("Division by zero is not allowed")
        return ALLOWED_BIN_OPS[type(node.op)](left_value, right_value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPS:
        return ALLOWED_UNARY_OPS[type(node.op)](_eval_arithmetic_ast(node.operand))

    raise ValueError("Expression contains unsupported operations")

def safe_calculator(expression: str) -> str:
    cleaned_expression = expression.strip()
    if not cleaned_expression:
        raise ValueError("Expression must not be empty")

    if len(cleaned_expression) > 120:
        raise ValueError("Expression too long")

    if not re.fullmatch(r"[0-9\s\+\-\*/\(\)\.]+", cleaned_expression):
        raise ValueError("Expression contains invalid characters")

    parsed = ast.parse(cleaned_expression, mode="eval")
    result = _eval_arithmetic_ast(parsed)

    if result.is_integer():
        return str(int(result))
    return str(result)

def calculator_tool(tool_input: str) -> str:
    return safe_calculator(tool_input)

def knowledge_base_tool(tool_input: str, k: int) -> str:
    if not chunks:
        return "No chunks available. Ingest documents first."

    results = retrieve_top_chunks(tool_input, k)
    if not results:
        return "No relevant chunks found."

    lines = []
    for item in results:
        lines.append(f"[{item['chunk_id']}] score={item['score']:.4f} :: {item['text']}")
    return "\n".join(lines)

def parse_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))

def fallback_agent_decision(query: str, steps: List[dict]) -> dict:
    lowered_query = query.lower()
    used_actions = {step["action"] for step in steps}
    has_math = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", query))
    knowledge_keywords = ["explain", "what is", "define", "why", "how", "recursion", "algorithm"]
    has_knowledge_signal = any(keyword in lowered_query for keyword in knowledge_keywords)

    if has_math and "calculator" not in used_actions:
        expression_match = re.search(r"[\d\s\+\-\*/\(\)\.]+", query)
        expression = expression_match.group(0).strip() if expression_match else query
        return {"action": "calculator", "input": expression}

    if has_knowledge_signal and "knowledge_base" not in used_actions:
        return {"action": "knowledge_base", "input": query}

    return {"action": "final", "input": "Provide final answer"}

def agent_decide_action(query: str, steps: List[dict]) -> dict:
    decision_system_prompt = (
        "You are an agent controller for tool use. "
        "You must return exactly one JSON object with keys: action and input. "
        "Allowed actions: calculator, knowledge_base, final. "
        "Choose calculator for arithmetic expressions. "
        "Choose knowledge_base for factual/conceptual questions that require retrieval. "
        "Choose final only when enough information is available to answer."
    )

    decision_user_prompt = (
        f"User query: {query}\n"
        f"Current steps: {json.dumps(steps, ensure_ascii=True)}\n"
        "Return JSON only. Example: {\"action\":\"calculator\",\"input\":\"25*48\"}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": decision_system_prompt},
                {"role": "user", "content": decision_user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        decision = parse_json_object(content)

        action = decision.get("action")
        tool_input = decision.get("input", "")
        if action not in {"calculator", "knowledge_base", "final"}:
            raise ValueError("Invalid action")
        if not isinstance(tool_input, str):
            raise ValueError("input must be a string")

        return {"action": action, "input": tool_input.strip()}
    except Exception:
        return fallback_agent_decision(query, steps)

def agent_generate_final_answer(query: str, steps: List[dict]) -> str:
    synthesis_prompt = (
        "You are a helpful CS teaching assistant. "
        "Synthesize a final answer using the available step outputs. "
        "If a step output says information is missing, explicitly mention the limitation."
    )

    user_prompt = (
        f"Query: {query}\n"
        f"Tool steps:\n{json.dumps(steps, ensure_ascii=True, indent=2)}\n"
        "Provide a concise final answer."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise ValueError("Model returned empty final answer")
    return answer

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

@app.get("/search", response_model=SearchResponse)
def search_chunks(query: str, k: int = 3):
    """
    Semantic search endpoint for debugging retrieval quality.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks available. Ingest documents first.")

    try:
        top_results = retrieve_top_chunks(cleaned_query, k)

        return {
            "query": cleaned_query,
            "results": top_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your search request")

@app.post("/qa", response_model=QAResponse)
def grounded_qa(request: QARequest):
    """
    Grounded QA endpoint.
    Uses retrieved chunks as context and returns citations.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks available. Ingest documents first.")

    conversation_history = sessions[request.session_id]

    try:
        top_chunks = retrieve_top_chunks(request.question, request.k)
        citations = [{"chunk_id": item["chunk_id"], "score": item["score"]} for item in top_chunks]

        context_blocks = []
        for item in top_chunks:
            context_blocks.append(
                f"[{item['chunk_id']}] (score={item['score']:.4f})\n{item['text']}"
            )
        context_text = "\n\n".join(context_blocks)

        grounding_instruction = (
            "Use only the context below to answer the question. "
            "If the answer is not in the context, say you do not have enough information from the ingested documents. "
            "When possible, mention chunk ids like [doc#1].\n\n"
            f"Context:\n{context_text}"
        )

        # Keep session history in natural chat form while injecting grounding context at generation time.
        conversation_history.append({"role": "user", "content": request.question})
        model_messages = conversation_history[:-1] + [
            {"role": "system", "content": grounding_instruction},
            {"role": "user", "content": request.question},
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=model_messages
        )

        answer = response.choices[0].message.content
        if not answer:
            raise HTTPException(status_code=500, detail="Model returned an empty response")

        conversation_history.append({"role": "assistant", "content": answer})
        turn_count = len([msg for msg in conversation_history if msg["role"] == "user"])

        return {
            "answer": answer,
            "citations": citations,
            "turn_count": turn_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing QA request: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your QA request")

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

@app.post("/agent", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    """
    LLM agent endpoint with tool use.
    Loop: decide -> act -> observe until final answer is produced.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    conversation_history = sessions[request.session_id]
    conversation_history.append({"role": "user", "content": request.query})

    steps: List[dict] = []
    max_steps = 6

    try:
        for _ in range(max_steps):
            decision = agent_decide_action(request.query, steps)
            action = decision["action"]
            tool_input = decision.get("input", "")

            if action == "final":
                final_answer = agent_generate_final_answer(request.query, steps)
                conversation_history.append({"role": "assistant", "content": final_answer})
                print(f"Agent session={request.session_id} completed with {len(steps)} step(s)")
                return {
                    "answer": final_answer,
                    "steps": steps,
                }

            if action == "calculator":
                try:
                    tool_output = calculator_tool(tool_input)
                except Exception as e:
                    tool_output = f"Calculator error: {str(e)}"
            elif action == "knowledge_base":
                tool_output = knowledge_base_tool(tool_input or request.query, request.k)
            else:
                tool_output = "Unsupported action"

            step = {
                "action": action,
                "input": tool_input,
                "output": tool_output,
            }
            steps.append(step)
            print(f"Agent step: {step}")

        final_answer = agent_generate_final_answer(request.query, steps)
        conversation_history.append({"role": "assistant", "content": final_answer})
        return {
            "answer": final_answer,
            "steps": steps,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing agent request: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your agent request")

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