from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db.database import engine, SessionLocal
from db import models
from routes import user
from routes import chat
from pydantic import BaseModel
import uvicorn
from utils.agentic_implementation import main as agent

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user.router)
app.include_router(chat.router)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "FastAPI with SQLite and SQLAlchemy"}

class ChatRequest(BaseModel):
    message: str

# This function will use the message as input.
def process_message(message: str) -> str:
    # For example, process the message (this could be any logic you need)
    return agent(message)

# New endpoint that gets user input from the body and processes it
@app.post("/processchat")
def process_chat(chat_request: ChatRequest):
    # Extract the message from the request body
    message = chat_request.message
    print(message)
    # Use the message as an input for another function
    result = process_message(message)
    # Return the processed result
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
    