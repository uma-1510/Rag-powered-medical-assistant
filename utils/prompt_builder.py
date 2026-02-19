def build_prompt(query, contexts, emergency_flag):

    context_text = ""
    citations = []

    for i, ctx in enumerate(contexts):
        context_text += f"[{i+1}] {ctx['text']}\n"

    warning = ""
    if emergency_flag:
        warning = """
If symptoms appear serious, strongly advise consulting a doctor immediately.
"""

    prompt = f"""
You are a careful medical assistant.

Rules:
- Answer ONLY using provided context
- Do NOT invent information
- Explain simply in plain English
- Provide helpful advice
- Mention when to see a doctor
- Cite sources using numbers
- If the answer is not found in context, reply: no found in context

Context:
{context_text}

User Question:
{query}

{warning}

Answer:
"""

    return prompt.strip()
