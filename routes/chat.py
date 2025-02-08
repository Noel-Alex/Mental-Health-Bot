from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import models, schemas


def get_db():
    from db.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter(prefix="/chat", tags=["Chats"])

@router.get("/getresponse")
def read_user():
    return {"message": "Chat response"}


