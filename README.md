# Mini-Chatbot App

## Assignment 2 Reflection Questions (Conversational API)

### 1. Why is conversation history necessary?
Conversation history gives the model context from previous turns, so it can understand follow-up questions, references, and user intent across multiple messages. Without history, each request is isolated and the assistant cannot maintain coherent multi-turn dialogue.

### 2. What happens if you remove the system prompt?
If the system prompt is removed, the assistant loses its behavior constraints (such as role, tone, and style). Responses may become less consistent, less focused on the assignment goal, and may vary more across turns.

### 3. How does context length affect performance and cost?
Longer context means more tokens are sent to the model each request. This increases latency and API cost, and can eventually hit model context limits. As chats grow, systems often trim or summarize old messages to balance quality, speed, and cost.

### 4. Why is in-memory storage not production safe?
In-memory storage is temporary and tied to one running process. Data is lost on restart or crash, and sessions are not shared across multiple server instances. This makes it unreliable for real users and difficult to scale.

### 5. What would change if you used a database?
With a database, session and message history would be persisted and shared across instances. You could recover data after restarts, support horizontal scaling, query historical records, and apply retention policies. The app would need CRUD logic, schema design, indexing, and migration/maintenance practices.

## Assignment 3 Reflection Questions (RAG Q&A Service)

### 7. Why does grounding reduce hallucinations?
Grounding reduces hallucinations because the model is instructed to answer using retrieved document chunks instead of relying only on parametric memory. By constraining generation to explicit context, the answer is tied to verifiable source text, which lowers the chance of fabricated facts. If the context is insufficient, a well-designed grounded prompt also tells the model to admit uncertainty.

### 8. How do chunk size and overlap affect retrieval quality?
Chunk size controls how much meaning is packed into each vector. Small chunks improve precision but may lose surrounding context; large chunks preserve context but may include irrelevant text that weakens similarity signals. Overlap helps preserve continuity across chunk boundaries, reducing information loss when important sentences are split. In practice, retrieval quality depends on balancing chunk size and overlap for the document style and question types.

### 9. What is the difference between semantic search and keyword search?
Keyword search matches exact terms or token patterns, so it works well when the query and documents use the same words. Semantic search compares vector embeddings, so it can retrieve relevant content even when wording differs (for example, synonyms or paraphrases). Keyword search is often faster and easier to explain, while semantic search is generally better at capturing intent and conceptual similarity.

### 10. What are common failure modes of RAG systems?
Common failure modes include poor chunking, weak retrieval recall, and prompt leakage where irrelevant chunks distract generation. Other risks are stale or low-quality source documents, embedding mismatch between queries and indexed content, and overly high k values that add noise. Even with retrieval, the model may still over-generalize unless prompts enforce grounded behavior and abstention when evidence is missing.

### 11. Why are citations important in AI systems?
Citations provide transparency by showing which evidence supports an answer. They improve trust, make outputs auditable, and help users quickly verify or challenge claims. Citations are also useful for debugging retrieval quality, because developers can inspect whether the right chunks were selected. In educational and professional settings, citations are essential for accountability and responsible AI use.

## Assignment 4 Reflection Questions (LLM Agent with Tool Use)

### 7. What is an LLM agent?
An LLM agent is a system where the model does more than generate one-shot text. It can decide whether a tool is needed, call that tool, observe the tool output, and then continue reasoning before producing a final answer. In this assignment, the agent behavior follows a loop: decide -> act -> observe.

### 8. Why not always call tools?
Tool calls increase latency, token usage, and implementation complexity. For simple requests, direct model responses are usually faster and sufficient, so calling tools every time is wasteful. Good agent design should call tools only when they improve correctness, such as exact arithmetic or retrieval-grounded knowledge.

### 9. What are risks of tool misuse?
Tool misuse can create security risks, incorrect outputs, and unstable behavior. For example, unsafe calculator execution can expose code injection risks, and wrong retrieval calls can introduce irrelevant context that harms answer quality. Excessive or incorrect tool calls also increase cost and can propagate errors through later reasoning steps.

### 10. How does prompt design affect decisions?
Prompt design strongly affects whether the agent chooses the right action. Clear instructions about available tools, when to use each tool, and strict output format constraints (for example, JSON action schemas) improve consistency. Ambiguous prompts often cause poor routing decisions, unnecessary tool calls, or premature final answers.

### 11. What are limitations of this agent?
This agent has several limitations. It only supports a small set of tools, so many tasks remain out of scope. Decision quality still depends on model reliability and prompt quality, and retrieval quality depends on chunking and embedding behavior. The in-memory session/chunk design is not persistent across restarts, and multi-step reasoning can still fail on ambiguous or complex queries.
