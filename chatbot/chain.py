from retriever.query_retriever import retrieve
from chatbot.generate_answer import generate

def rag(query):
    docs = retrieve(query)
    return generate(query, docs)
