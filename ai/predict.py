import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics

from train import ranges, Test, Train  
from Processing import Processing

class Experiment:
        def __init__(self, processing_params=None, test_params=None):
                """
                This is a class containing the full experiment pipeline.
                It takes in a dataset, allows user choices of:
                        - models
                        - features
                        - target
                and gives back a class which can be used to predict new data.
                """
                self.processing_params = processing_params if processing_params else {
                        'target_column': 'target',  # target variable name
                        'dropNAMethod': 'drop',  # options: 'mean', 'median', 'mode', 'drop'
                        'fix imbalance': 'oversample', # options: 'smote', 'undersample', 'oversample', None
                        'scaling_method': 'standard',  # options: 'standard', 'minmax', None 
                        'apply_pca': True, 
                        'pca_method': 'auto', #options: {'full', 'arpack', 'auto', 'covariance_eigh', 'randomized'}
                        'pca_components': 5, #pca param
                        'text_vectorization': 'tfidf',  # options: 'tfidf', 'count', 'onehot'
                        'random_state': 42
                }
                self.test_params = test_params if test_params else {
                        'classification': True,
                        'scoreFunc': metrics.accuracy_score,
                }

                self.processor = None
                self.model = None
                self.model_name = None
                self.score_val = None
        
        def run(self, data: pd.DataFrame):
                """
                Runs the full experiment: Processing -> Model Selection -> Final Training.
                """
                print("Starting Experiment...")
                
                # 1. Initialize and Run Processing
                print("Step 1: Processing Data...")
                self.processor = Processing(data, self.processing_params)
                processed_data = self.processor.train_processing_pipeline()
                X, y = processed_data.X, processed_data.y
                
                print(f"Data Processed. Features shape: {X.shape}")

                # 2. Run Model Selection (Testing)
                print("Step 2: Testing Models...")
                test = Test(classification=self.test_params['classification'], 
                           scoreFunc=self.test_params['scoreFunc'])
                
                # fullTests splits data internally for validation
                best_modelFunc, best_modelName, best_score, scoreboard = test.fullTests(X, y)
                
                self.model_name = best_modelName
                self.score_val = best_score
                print(f"Best Model Selected: {best_modelName} with Validation Score: {best_score}")

                # 3. Train Final Model
                # We want to train on as much data as possible, or stick to the Train class logic.
                # The user requested "single processing pipeline, tester and trainer".
                # We'll use the Train class but maybe we should train on ALL X/y?
                # The Train class does another split. Let's stick to using Train class for consistency 
                # with existing codebase but essentially we might want to override.
                # For now, let's just train the model instance on X and y directly to maximize data usage for final model.
                
                print("Step 3: Training Final Model...")
                self.model = best_modelFunc()
                self.model.fit(X, y)
                print("Final Model Trained.")
                
                return self.score_val

        def predict(self, X: pd.DataFrame):
                """
                Predicts on new data using the fitted processor and model.
                """
                if self.processor is None or self.model is None:
                        raise ValueError("Experiment has not been run yet. Call run() first.")
                
                # 1. Process new data (Transform only)
                X_processed = self.processor.transform(X)
                
                # 2. Predict
                predictions = self.model.predict(X_processed)
                return predictions

        def save(self, filename="experiment.pkl"):
                """
                Saves the entire Experiment object.
                """
                with open(filename, 'wb') as f:
                        pickle.dump(self, f)
                print(f"Experiment saved to {filename}")

        @staticmethod
        def load(filename="experiment.pkl"):
                """
                Loads an Experiment object.
                """
                with open(filename, 'rb') as f:
                        experiment = pickle.load(f)
                return experiment

if __name__ == "__main__":
    # Test on Iris
    from sklearn.datasets import load_iris
    
    #pick another similiar dataset 
    from sklearn.datasets import load_wine

    # Load data
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into Train (Experiment) and Unseen (Predict)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].drop(columns=['target'])
    test_y = df.iloc[split_idx:]['target']
    
    # Define Params
    proc_params = {
        'target_column': 'target',
        'dropNAMethod': 'drop',
        'fix imbalance': None,
        'scaling_method': None,
        'apply_pca': False,
        'pca_components': 10,
        'text_vectorization': 'tfidf',
        'random_state': 42
    }
    
    exp = Experiment(processing_params=proc_params)
    exp.run(train_df)
    
    print("\n--- Inference Test ---")
    predictions = exp.predict(test_df)
    print("Predictions:", predictions)
    print("Actuals:    ", test_y.values)
    
    from sklearn.metrics import accuracy_score
    print("Inference Accuracy:", accuracy_score(test_y, predictions))

