# src/qa.py
import os
import openai

openai.api_key = os.getenv("GEMINI_API_KEY")

def ask_question(chunks, question):
    context = "\n".join(chunks)
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    response = openai.ChatCompletion.create(
        model="gemini-1",  # change if using another Gemini model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message['content']
    return answer