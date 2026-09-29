# src/app.py

from flask import Flask, request, render_template_string
from dotenv import load_dotenv
import os

from retrieval import get_relevant_chunks
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

HTML = """
<!doctype html>
<title>Personalized Learning Assistant</title>
<h2>Ask a Question</h2>
<form method="post">
    <input name="question" style="width:400px">
    <input type="submit">
</form>
{% if answer %}
<h3>Answer:</h3>
<p>{{ answer }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def ask():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        context = "\n".join(get_relevant_chunks(question))

        prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
        response = llm.invoke(prompt)
        answer = response.content

    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
