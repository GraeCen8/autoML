#this is for database models NOT ai models and ml models 

from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
from pydantic import BaseModel

#this uses and implements all the models in plan.Md 


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password: str
    email: str

class Database(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    name: str
    file_path: str

class Experiment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    database_id: int = Field(foreign_key="database.id")
    name: str
    target: str
    features: str
    models: str
    status: str = Field(default="pending") # pending, running, completed, failed
    model_path: Optional[str] = None

class Model(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id")
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    model_path: Optional[str] = None # Changed from bytes to path logic if needed, but keeping simple for now

class Result(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id")
    model_id: Optional[int] = Field(default=None, foreign_key="model.id") # Made optional as we might just save experiment result
    accuracy: float
    precision: float
    recall: float
    f1: float
    model_path: Optional[str] = None
