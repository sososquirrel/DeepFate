from sklearn.base import BaseEstimator, TransformerMixin
import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import pandas as pd
import numpy as np
from DeepFate.models.utils_models import remove_highly_correlated_features



class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9,priority_vars=[], blacklist_vars=[]):
        self.threshold = threshold
        self.priority_vars = priority_vars
        self.blacklist_vars = blacklist_vars

    def fit(self, X, y=None):
        # Identify features to drop based on correlation threshold
        self.to_drop_ =remove_highly_correlated_features(train_dataset=X, 
                               corr_threshold=self.threshold, 
                               priority_vars=self.priority_vars,
                               blacklist_vars = self.blacklist_vars)  # Set to store columns to drop

        return self  # Return self to allow for chaining

    def transform(self, X):
        # Convert the input to a pandas DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Drop the correlated features from the data
        X_transformed = X.drop(columns=list(self.to_drop_))

        # Return the transformed data as a DataFrame
        return X_transformed

    def get_support(self):
        # Return the set of columns that were dropped
        return self.to_drop_
