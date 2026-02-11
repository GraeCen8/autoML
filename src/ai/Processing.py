import pandas as pd 
import numpy as np
import os

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from sklearn.model_selection import train_test_split

user_params = {
    'target_column': 'target',    # target variable name
    'scaling_method': 'standard',  # options: 'standard', 'minmax', None
    'apply_pca': False,
    'pca_method': 'svd', #pca param
    'pca_components': 10, #pca param
    'text_vectorization': 'tfidf',  # options: 'tfidf', 'count', 'onehot
    'val_size': 0.2,
    'test_size': 0.1,
    'random_state': 42
}


class Data():
    def __init__(self, X, y, vectorizer, scaler, pca) -> None:
        self.X = X
        self.y = y
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.pca = pca


class Processing:
    def __init__(self, data: pd.DataFrame, user_params: dict):
        """
        takes in data and user preferences for processing.
        Initializes the processing class with default settings.
        Parameters:
        data (pd.DataFrame): The input data to be processed.
        user_params (dict): A dictionary of user preferences for processing.
        """
        self.data = data
        self.params = user_params
        
        self.vectorizers = {} # Dictionary to store vectorizers per column
        self.scaler = None
        self.pca = None
        self.imputation_values = {} # Store values for NA filling
        
    def fit_transform_drop_na(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        method = self.params.get('dropNAMethod', 'drop')
    
        # X dropping NA
        if method == 'mean':
            self.imputation_values = X.mean(numeric_only=True).to_dict()
            X = X.fillna(self.imputation_values)
        elif method == 'median':
            self.imputation_values = X.median(numeric_only=True).to_dict()
            X = X.fillna(self.imputation_values)
        elif method == 'mode':
            # mode() returns a DataFrame, take first row
            mode_df = X.mode()
            if not mode_df.empty:
                self.imputation_values = mode_df.iloc[0].to_dict()
                X = X.fillna(self.imputation_values)
        elif method == 'drop':
            # Need to align X and y after dropping
            valid_indices = X.dropna().index
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
        elif method == '0':
            X = X.fillna(0)
        elif method == '1':
            X = X.fillna(1)
    
        # y dropping NA and align with X
        # For y, we generally just drop NAs or align with X
        if y is not None:
             valid_y_indices = y.dropna().index
             common_indices = X.index.intersection(valid_y_indices)
             X = X.loc[common_indices]
             y = y.loc[common_indices]
            
        return X, y

    def transform_drop_na(self, X: pd.DataFrame) -> pd.DataFrame:
        method = self.params.get('dropNAMethod', 'drop')
        
        if method in ['mean', 'median', 'mode']:
            if self.imputation_values:
                X = X.fillna(self.imputation_values)
        elif method == 'drop':
             X = X.dropna()
        elif method == '0':
            X = X.fillna(0)
        elif method == '1':
             X = X.fillna(1)
             
        return X
    
    def fix_imbalance(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling technique to address class imbalance.
        Only applied during training (fit_transform).
        """
        method = self.params.get('fix imbalance', None)
        
        if method is None or method not in ['smote', 'undersample', 'oversample']:
            return X, y
        
        random_state = self.params.get('random_state', 42)
        
        # Select and apply resampler
        if method == 'smote':
            resampler = SMOTE(random_state=random_state)
        elif method == 'undersample':
            resampler = RandomUnderSampler(random_state=random_state)
        else:  # oversample
            resampler = RandomOverSampler(random_state=random_state)
        
        # Resample data
        # Note: resample converts to numpy array, losing column names
        try:
             result = resampler.fit_resample(X, y)
             X_res, y_res = result[0], result[1]
             
             # Ensure y_res is a 1D Series
             if isinstance(y_res, pd.DataFrame):
                 y_res = y_res.iloc[:, 0]
             
             # Return as DataFrame and Series with preserved column names
             return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception as e:
            print(f"Resampling failed: {e}. Continuing without resampling.")
            return X, y
    
    def fit_transform_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorize text columns in the dataframe and store vectorizers.
        """
        text_cols = data.select_dtypes(include=['object']).columns
        # Also include columns that might be text but not skipped by other checks if needed, 
        # but select_dtypes(include=['object']) is standard for pandas strings.
        
        for col in text_cols:
            if self.params.get('text_vectorization') == 'tfidf':
                vectorizer = TfidfVectorizer()
            elif self.params.get('text_vectorization') == 'count':
                vectorizer = CountVectorizer()
            elif self.params.get('text_vectorization') == 'onehot':
                vectorizer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            else:
                continue
            
            # Fit and transform
            if self.params.get('text_vectorization') == 'onehot':
                 transformed_data = vectorizer.fit_transform(data[[col]]) # OneHot expects 2D
                 cols = vectorizer.get_feature_names_out([col])
                 vectorized_df = pd.DataFrame(
                    transformed_data, 
                    columns=cols,
                    index=data.index
                )
            else:
                 transformed_data = vectorizer.fit_transform(data[col])
                 vectorized_df = pd.DataFrame(
                    transformed_data.toarray(), 
                    columns=[f"{col}_{i}" for i in range(transformed_data.shape[1])],
                    index=data.index
                )
            
            self.vectorizers[col] = vectorizer
            data = pd.concat([data.drop(columns=[col]), vectorized_df], axis=1)
            
        return data

    def transform_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorize using fitted vectorizers.
        """
        # We need to iterate over the columns that we have vectorizers for.
        # However, if the new data doesn't have the column, we skip.
        
        for col, vectorizer in self.vectorizers.items():
            if col not in data.columns:
                continue
                
            if self.params.get('text_vectorization') == 'onehot':
                 transformed_data = vectorizer.transform(data[[col]])
                 cols = vectorizer.get_feature_names_out([col])
                 vectorized_df = pd.DataFrame(
                    transformed_data, 
                    columns=cols,
                    index=data.index
                )
            else:
                 transformed_data = vectorizer.transform(data[col])
                 vectorized_df = pd.DataFrame(
                    transformed_data.toarray(), 
                    columns=[f"{col}_{i}" for i in range(transformed_data.shape[1])],
                    index=data.index
                )
            
            data = pd.concat([data.drop(columns=[col]), vectorized_df], axis=1)
            
        return data


    def fit_transform_scale(self, data: pd.DataFrame) -> pd.DataFrame: 
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
            
        if self.params.get('scaling_method') == 'standard':
            self.scaler = StandardScaler()
        elif self.params.get('scaling_method') == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return data
        
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        return data

    def transform_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            return data
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Ensure we only try to scale columns that were seen during fit? 
        # The scaler expects specific feature count. 
        # Usually we assume the schema (numeric cols) is same.
        
        try:
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        except Exception as e:
            print(f"Scaling failed: {e}. Returning unscaled data.")
            
        return data
        
    def fit_transform_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
            
        if self.params.get('apply_pca'):
            n_components = min(self.params.get('pca_components', 5), len(numeric_cols), len(data))
            self.pca = PCA(n_components=n_components, svd_solver=self.params.get('pca_method', 'auto'))
            pca_data = self.pca.fit_transform(data[numeric_cols])
            
            # Replace numeric columns with PCA components
            pca_df = pd.DataFrame(
                pca_data,
                columns=[f'PC{i+1}' for i in range(pca_data.shape[1])],
                index=data.index
            )
            data = pd.concat([data.drop(columns=numeric_cols), pca_df], axis=1)
            return data
        else:
            return data

    def transform_pca(self, data: pd.DataFrame) -> pd.DataFrame:
         if self.pca is None:
             return data
         
         numeric_cols = data.select_dtypes(include=[np.number]).columns
         if len(numeric_cols) == 0:
            return data
            
         pca_data = self.pca.transform(data[numeric_cols])
         pca_df = pd.DataFrame(
                pca_data,
                columns=[f'PC{i+1}' for i in range(pca_data.shape[1])],
                index=data.index
            )
         data = pd.concat([data.drop(columns=numeric_cols), pca_df], axis=1)
         return data
    
    def split_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        target_col = self.params['target_column']
        if target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
             # If target no in data, assume it's X only data or raise error?
             # For training, we need target.
            raise ValueError(f"Target column '{target_col}' not found in data.")
        return X, y
    
    def train_processing_pipeline(self) -> Data:
        """
        Runs the full processing pipeline for training data (fit_transform).
        """
        X, y = self.split_target(self.data)
        X, y = self.fit_transform_drop_na(X, y)
        X = self.fit_transform_text(X)
        X, y = self.fix_imbalance(X, y) # Only valid for training
        X = self.fit_transform_scale(X)
        X = self.fit_transform_pca(X)
        
        out = Data(X, y, self.vectorizers, self.scaler, self.pca)
        return out
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted processing pipeline to new data (transform only).
        """
        # Ensure we work on a copy to avoid side effects
        X = X.copy()
        
        # 1. Drop NA (using stats from fit if applicable)
        X = self.transform_drop_na(X)
        
        # 2. Text Vectorization
        X = self.transform_text(X)
        
        # 3. Imbalance - SKIP for inference
        
        # 4. Scaling
        X = self.transform_scale(X)
        
        # 5. PCA
        X = self.transform_pca(X)
        
        return X


if __name__ == "__main__": 
    user_params = {
        'target_column': 'target',  # target variable name
        'dropNAMethod': 'drop',  # options: 'mean', 'median', 'mode', 'drop'
        'fix imbalance': 'oversample', # options: 'smote', 'undersample', 'oversample', None
        'scaling_method': 'standard',  # options: 'standard', 'minmax', None 
        'apply_pca': True, 
        'pca_method': 'auto', #options: {'full', 'arpack', 'auto', 'covariance_eigh', 'randomized'}
        'pca_components': 2, #pca param
        'text_vectorization': 'tfidf',  # options: 'tfidf', 'count', 'onehot'
        'random_state': 42
    }
    
    # Example usage
    df = pd.DataFrame({
        'text': ['this is a sample', 'another sample text', 'more text data', 'sample'],
        'num1': [1, 2, None, 4],
        'num2': [4, 5, 6, 7],
        'target': [0, 1, 0, 1]
    })
    
    print("Original Data:\n", df)
    
    processor = Processing(df, user_params)
    data = processor.train_processing_pipeline()
    
    print("\nProcessed Training Data (X):\n", data.X)
    print("\nProcessed Training Data (y):\n", data.y)
    
    # Test Inference
    new_data = pd.DataFrame({
         'text': ['sample text', 'new data'],
         'num1': [2, 3],
         'num2': [5, 6]
    })
    print("\nNew Data:\n", new_data)
    processed_new = processor.transform(new_data)
    print("\nProcessed New Data:\n", processed_new)