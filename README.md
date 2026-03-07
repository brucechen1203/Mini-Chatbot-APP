# Mini-Chatbot App

## Reflection Questions

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
