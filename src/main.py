from sqlmodel import SQLModel, Field, create_engine, Session, select
from fastapi import FastAPI, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import shutil
import os
import uuid

#import our stuff
from database import create_tables, get_session, engine
from models import User, Database, Experiment, Model, Result
from worker import run_experiment_task

# Create tables on startup
create_tables()

app = FastAPI()

# Enable CORS for frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the autoML API"}

#
# Auth Routes (Simple)
#
class LoginRequest(BaseModel):
    username: str

@app.post("/login")
def login(request: LoginRequest, session: Session = Depends(get_session)):
    # Check if user exists, if not create
    statement = select(User).where(User.username == request.username)
    user = session.exec(statement).first()
    
    if not user:
        user = User(username=request.username, password="password", email=f"{request.username}@example.com") # Dummy defaults
        session.add(user)
        session.commit()
        session.refresh(user)
    
    return user

#
# Database Routes (with File Upload)
#

@app.post("/database")
def create_database(
    name: str = Form(...),
    user_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    # verification that user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # save file
    file_extension = file.filename.split(".")[-1]
    if file_extension != "csv":
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    file_path = f"uploads/{uuid.uuid4()}.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_entry = Database(name=name, user_id=user_id, file_path=file_path)
    session.add(db_entry)
    session.commit()
    session.refresh(db_entry)
    return db_entry

@app.get("/database/{database_id}")
def read_database(database_id: int, session: Session = Depends(get_session)):
    database = session.get(Database, database_id)
    if not database:
        raise HTTPException(status_code=404, detail="Database not found")
    return database

@app.get("/user/{user_id}/databases")
def get_user_databases(user_id: int, session: Session = Depends(get_session)):
    statement = select(Database).where(Database.user_id == user_id)
    databases = session.exec(statement).all()
    return databases

#
# Experiment Routes
#

@app.post("/experiment")
def create_experiment(experiment: Experiment, session: Session = Depends(get_session)):
    # Validate database exists
    db = session.get(Database, experiment.database_id)
    if not db:
        raise HTTPException(status_code=404, detail="Database not found")

    experiment.status = "pending"
    session.add(experiment)
    session.commit()
    session.refresh(experiment)
    return experiment

@app.post("/experiment/{experiment_id}/run")
def run_experiment(
    experiment_id: int, 
    background_tasks: BackgroundTasks, 
    session: Session = Depends(get_session)
):
    experiment = session.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    experiment.status = "queued"
    session.add(experiment)
    session.commit()
    
    background_tasks.add_task(run_experiment_task, experiment_id)
    return {"message": "Experiment started", "status": "queued"}

@app.get("/experiment/{experiment_id}")
def get_experiment(experiment_id: int, session: Session = Depends(get_session)):
    experiment = session.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

@app.get("/database/{database_id}/experiments")
def get_database_experiments(database_id: int, session: Session = Depends(get_session)):
    statement = select(Experiment).where(Experiment.database_id == database_id)
    experiments = session.exec(statement).all()
    return experiments

@app.get("/experiment/{experiment_id}/results")
def get_experiment_results(experiment_id: int, session: Session = Depends(get_session)):
    statement = select(Result).where(Result.experiment_id == experiment_id)
    results = session.exec(statement).all()
    return results

@app.delete("/database/{database_id}")
def delete_database(database_id: int, session: Session = Depends(get_session)):
    database = session.get(Database, database_id)
    if not database:
        raise HTTPException(status_code=404, detail="Database not found")
    session.delete(database)
    session.commit()
    return {"ok": True}


def main():
    create_tables()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


