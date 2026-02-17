**# Conversation State Schema — RAEC**



**## Purpose**



**Provide session-level continuity without collapsing RAEC into a chatbot.**



**This layer must:**



**- Preserve thread awareness**

**- Track active tasks**

**- Resolve references**

**- Maintain ergonomic continuity**

**- Remain separate from cognitive core**



**---**



**## Data Structure**



**```python**

**class ConversationState:**

    **session\_id: str**

    **created\_at: datetime**

    **last\_updated: datetime**

    

    **active\_thread\_id: str**

    **mode: Literal\["chat", "task", "analysis", "meta"]**



    **active\_task: Optional\[str]**

    **unresolved\_references: List\[str]**

    **last\_commitments: List\[str]**



    **rolling\_summary: str**

    **recent\_turns: List\[Turn]  # last N raw turns**



    **def update\_from\_turn(user\_input: str, assistant\_output: str) -> None:**

        **...**



    **def generate\_prompt\_context() -> str:**

        **...**



    **def compress\_history() -> None:**

        **...**



