import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


### custom transformers to encode category columns
class CatAttrEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
    
        # Encode ordinal categorical features
        X['experience_level'] = X['experience_level'].replace({'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4})
        X['company_size'] = X['company_size'].replace({'S': 1, 'M': 2, 'L': 3})

        # Trim spaces from employment type
        X['employment_type'] = X['employment_type'].str.strip()

        # Convert binary categorical features
        X['fulltime_emp'] = (X['employment_type'] == 'FT').astype(int)
        X['us_currency'] = (X['salary_currency'] == 'USD').astype(int)
        X['us_emp_residence'] = (X['employee_residence'] == 'US').astype(int)
        X['us_company_loc'] = (X['company_location'] == 'US').astype(int)

        return X[['experience_level', 'company_size', 'fulltime_emp', 'us_currency', 'us_emp_residence', 'us_company_loc']]
    
    
class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
    
        # Clalcuate years since 2020
        X['work_year'] = X['work_year'] - 2020
        
        return X[['work_year']]
    
    
time_num_pipeline = Pipeline([
     ('year_transformer', YearTransformer()),
     ('std_scaler', StandardScaler()),
 ])