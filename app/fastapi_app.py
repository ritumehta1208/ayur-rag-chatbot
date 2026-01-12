from fastapi import FastAPI
from chatbot.chain import rag

app = FastAPI()

@app.get("/ask")
def ask(q: str):
    return {"answer": rag(q)}
