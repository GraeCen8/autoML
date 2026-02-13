import os
import pickle
import pandas as pd
from sqlmodel import Session, select, create_engine
from database import get_session, engine
from models import Experiment as ExperimentDB, Result, Database
from ai.predict import Experiment as ExperimentLogic

# Ensure directories exist
os.makedirs("learned_models", exist_ok=True)

def run_experiment_task(experiment_id: int):
    print(f"Worker: Starting experiment {experiment_id}")
    
    with Session(engine) as session:
        # 1. Fetch Experiment and Database info
        experiment_db = session.get(ExperimentDB, experiment_id)
        if not experiment_db:
            print(f"Worker: Experiment {experiment_id} not found")
            return

        database_db = session.get(Database, experiment_db.database_id)
        if not database_db:
             print(f"Worker: Database for experiment {experiment_id} not found")
             experiment_db.status = "failed"
             session.add(experiment_db)
             session.commit()
             return

        # Update status to running
        experiment_db.status = "running"
        session.add(experiment_db)
        session.commit()

        try:
            # 2. Load Data
            file_path = database_db.file_path
            print(f"Worker: Loading data from {file_path}")
            
            # Basic CSV loading - extend logic if needed for other formats
            df = pd.read_csv(file_path)
            
            # 3. Setup Experiment Logic
            # Parse features/target from DB if needed, or use logic class defaults
            # For now, we assume the user might have passed some params, but the current ExperimentLogic
            # is designed to take params in __init__. 
            # We'll use the target from the DB record.
            
            processing_params = {
                'target_column': experiment_db.target,
                'dropNAMethod': 'drop',
                'fix imbalance': 'oversample',
                'scaling_method': 'standard',
                'apply_pca': True,
                'pca_components': 5,
                'text_vectorization': 'tfidf',
                'random_state': 42
            }
            
            # Initialize Logic
            exp_logic = ExperimentLogic(processing_params=processing_params)
            
            # 4. Run Experiment
            print("Worker: Running experiment logic...")
            score = exp_logic.run(df)
            
            # 5. Save Artifacts
            model_filename = f"learned_models/experiment_{experiment_id}.pkl"
            exp_logic.save(model_filename)
            
            # 6. Save Results to DB
            result = Result(
                experiment_id=experiment_id,
                model_id=None, # storing the whole experiment logic for now
                accuracy=score if experiment_db.target else 0.0, # Placeholder mapping
                precision=0.0, # Placeholder
                recall=0.0,    # Placeholder
                f1=0.0,        # Placeholder
                model_path=model_filename
            )
            session.add(result)
            
            # 7. Update Experiment Status
            experiment_db.status = "completed"
            experiment_db.model_path = model_filename
            session.add(experiment_db)
            session.commit()
            
            print(f"Worker: Experiment {experiment_id} completed successfully.")

        except Exception as e:
            print(f"Worker: Experiment {experiment_id} failed: {e}")
            experiment_db.status = "failed"
            session.add(experiment_db)
            session.commit()
