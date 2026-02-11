from sqlmodel import SQLModel, Field, create_engine, Session
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

#database setup
DATABASE_URL = "sqlite:///app.db"
engine = create_engine(DATABASE_URL)

def create_tables():
    SQLModel.metadata.create_all(engine)
def get_session():
    with Session(engine) as session:
        yield session


#fastapi app setup
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the autoML API"}






def main():
    create_tables()
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    main()


