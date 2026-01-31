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
        
        self.vectorizer = None
        self.scaler = None
        self.pca = None
        
    def drop_na(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        method = self.params['dropNAMethod']
    
        # X dropping NA
        if method == 'mean':
            X = X.fillna(X.mean(numeric_only=True))
        elif method == 'median':
            X = X.fillna(X.median(numeric_only=True))
        elif method == 'mode':
            X = X.fillna(X.mode().iloc[0])
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
        valid_y_indices = y.dropna().index
        common_indices = X.index.intersection(valid_y_indices)
        X = X.loc[common_indices]
        y = y.loc[common_indices]
            
        return X, y
    
    def fix_imbalance(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling technique to address class imbalance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of resampled (X, y)
        """
        method = self.params.get('fix imbalance', None)
        
        if method is None or method not in ['smote', 'undersample', 'oversample']:
            return X, y
        
        random_state = self.params['random_state']
        
        # Select and apply resampler
        if method == 'smote':
            resampler = SMOTE(random_state=random_state)
        elif method == 'undersample':
            resampler = RandomUnderSampler(random_state=random_state)
        else:  # oversample
            resampler = RandomOverSampler(random_state=random_state)
        
        # Resample data
        result = resampler.fit_resample(X, y)
        X_res, y_res = result[0], result[1]
        
        # Ensure y_res is a 1D Series
        if isinstance(y_res, pd.DataFrame):
            y_res = y_res.iloc[:, 0]
        
        # Return as DataFrame and Series with preserved column names
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    
    def text_vectorize(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Vectorize text columns in the dataframe.
        
        Returns:
            Tuple of (transformed_data, list_of_vectorizers)
        """
        text_cols = data.select_dtypes(include=['object']).columns
        vectorizers = []
        
        for col in text_cols:
            if self.params['text_vectorization'] == 'tfidf':
                vectorizer = TfidfVectorizer()
                transformed_data = vectorizer.fit_transform(data[col])
                vectorized_df = pd.DataFrame(
                    transformed_data.toarray(), 
                    columns=[f"{col}_{i}" for i in range(transformed_data.shape[1])],
                    index=data.index
                )
            elif self.params['text_vectorization'] == 'count':
                vectorizer = CountVectorizer()
                transformed_data = vectorizer.fit_transform(data[col])
                vectorized_df = pd.DataFrame(
                    transformed_data.toarray(), 
                    columns=[f"{col}_{i}" for i in range(transformed_data.shape[1])],
                    index=data.index
                )
            elif self.params['text_vectorization'] == 'onehot':
                vectorizer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                transformed_data = vectorizer.fit_transform(data[[col]])
                vectorized_df = pd.DataFrame(
                    transformed_data, 
                    columns=[f"{col}_{i}" for i in range(transformed_data.shape[1])],
                    index=data.index
                )
            else:
                continue
            
            vectorizers.append(vectorizer)
            data = pd.concat([data.drop(columns=[col]), vectorized_df], axis=1)
            
        return data, vectorizers
            
    def scale_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | None]: 
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, None
            
        if self.params['scaling_method'] == 'standard':
            scaler = StandardScaler()
        elif self.params['scaling_method'] == 'minmax':
            scaler = MinMaxScaler()
        else:
            return data, None
        
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data, scaler
        
    def apply_pca(self, data: pd.DataFrame) -> tuple[pd.DataFrame, PCA | None]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, None
            
        if self.params['apply_pca']:
            n_components = min(self.params['pca_components'], len(numeric_cols), len(data))
            pca = PCA(n_components=n_components, svd_solver=self.params['pca_method'])
            pca_data = pca.fit_transform(data[numeric_cols])
            
            # Replace numeric columns with PCA components
            pca_df = pd.DataFrame(
                pca_data,
                columns=[f'PC{i+1}' for i in range(pca_data.shape[1])],
                index=data.index
            )
            data = pd.concat([data.drop(columns=numeric_cols), pca_df], axis=1)
            return data, pca
        else:
            return data, None
    
    def split_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        target_col = self.params['target_column']
        if target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        return X, y
    
    def train_processing_pipeline(self) -> Data:
        X, y = self.split_target(self.data)
        X, y = self.drop_na(X, y)
        X, vectorizers = self.text_vectorize(X)
        X, y = self.fix_imbalance(X, y)
        X, scaler = self.scale_data(X)
        X, pca = self.apply_pca(X)
        out = Data(X, y, vectorizers, scaler, pca)
        
        return out


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
        'val_size': 0.2,
        'test_size': 0.1,
        'random_state': 42
    }
    
    # Example usage
    df = pd.DataFrame({
        'text': ['this is a sample', 'another sample text', 'more text data'],
        'num1': [1, 2, None],
        'num2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    processor = Processing(df, user_params)
    processed_data = processor.train_processing_pipeline()
    
    print("Processed Features:\n", processed_data.X)
    print("\nProcessed Target:\n", processed_data.y)
    print("\nVectorizers:\n", processed_data.vectorizer)
    print("\nScaler:\n", processed_data.scaler)
    print("\nPCA:\n", processed_data.pca)