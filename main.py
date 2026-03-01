from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from googletrans import Translator
import torch
import json

query_log = []

app = FastAPI()
translator = Translator()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load Knowledge Base
# -------------------------------

def load_knowledge():
    global knowledge, questions, answers, question_embeddings
    with open("knowledge.json", "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    questions = [item["question"] for item in knowledge]
    answers = [item["answer"] for item in knowledge]

    question_embeddings = model.encode(questions, convert_to_tensor=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
load_knowledge()

# -------------------------------
# CHAT API
# -------------------------------

@app.get("/chat")
def chat(user_question: str):
    try:
        detected_lang = detect(user_question)
        if detected_lang not in ["en", "hi", "kn"]:
            detected_lang = "en"
    except:
        detected_lang = "en"

    if detected_lang != "en":
        user_question = translator.translate(user_question, dest="en").text

    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, question_embeddings)

    score = torch.max(similarity).item()
    index = torch.argmax(similarity).item()

    if score < 0.6:
        response = "Sorry, I could not understand your question. Please contact the college office for more details."
    else:
        response = answers[index]

    if detected_lang != "en":
        response = translator.translate(response, dest=detected_lang).text


    query_log.append({
    "question": user_question,
    "confidence": score
})
    
    return {
        "response": response,
        "confidence_score": float(score)
    }

# -------------------------------
# ADMIN AUTH (Simple Login)
# -------------------------------

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

logged_in = False

@app.get("/admin-login", response_class=HTMLResponse)
def admin_login_page():
    return """
    <h2>Admin Login</h2>
    <form method="post">
        Username: <input type="text" name="username"><br><br>
        Password: <input type="password" name="password"><br><br>
        <button type="submit">Login</button>
    </form>
    """

@app.post("/admin-login")
def admin_login(username: str = Form(...), password: str = Form(...)):
    global logged_in
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        logged_in = True
        return RedirectResponse("/admin", status_code=303)
    return "Invalid credentials"

# -------------------------------
# ADMIN DASHBOARD
# -------------------------------

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    if not logged_in:
        return RedirectResponse("/admin-login", status_code=303)

    html = "<h2>Admin Dashboard</h2>"

    html += """
    <h3>Add New Q&A</h3>
    <form method="post" action="/add">
        Question:<br>
        <input type="text" name="question" style="width:400px"><br><br>
        Answer:<br>
        <input type="text" name="answer" style="width:400px"><br><br>
        <button type="submit">Add</button>
    </form>
    """

    html += "<h3>Existing Q&A</h3><ul>"
    for idx, item in enumerate(knowledge):
        html += f"<li>{item['question']} - <a href='/delete/{idx}'>Delete</a></li>"
    html += "</ul>"

    return html

# -------------------------------
# ADD NEW Q&A
# -------------------------------

@app.post("/add")
def add_question(question: str = Form(...), answer: str = Form(...)):
    if not logged_in:
        return RedirectResponse("/admin-login", status_code=303)

    knowledge.append({"question": question, "answer": answer})

    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=4)

    load_knowledge()

    return RedirectResponse("/admin", status_code=303)

# -------------------------------
# DELETE Q&A
# -------------------------------

@app.get("/delete/{index}")
def delete_question(index: int):
    if not logged_in:
        return RedirectResponse("/admin-login", status_code=303)

    knowledge.pop(index)

    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=4)

    load_knowledge()

    return RedirectResponse("/admin", status_code=303)
@app.get("/analytics", response_class=HTMLResponse)
def analytics_dashboard():
    from collections import Counter

    total_queries = len(query_log)

    question_counts = Counter([q["question"] for q in query_log])
    top_questions = question_counts.most_common(5)

    rejected = len([q for q in query_log if q["confidence"] < 0.6])

    html = f"""
    <h2>Chatbot Analytics Dashboard</h2>
    <p><b>Total Queries:</b> {total_queries}</p>
    <p><b>Rejected Queries:</b> {rejected}</p>

    <h3>Top 5 Most Asked Questions</h3>
    <ul>
    """

    for question, count in top_questions:
        html += f"<li>{question} ({count} times)</li>"

    html += "</ul>"

    return html