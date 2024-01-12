from os import X_OK
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str, date_var:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")
        if not isinstance(date_var, str):
            raise ValueError("date variable name should be a string")

        self.variable = variable
        self.date_var = date_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.date_var] = pd.to_datetime(X[self.date_var], format='%Y-%m-%d')
        wkday_null_idx = X[X[self.variable].isnull() == True].index
        X.loc[wkday_null_idx, self.variable] = X.loc[wkday_null_idx, self.date_var].dt.day_name().apply(lambda x: x[:3])

        # drop 'dteday' column after imputation
        X.drop(self.date_var, axis=1, inplace=True)

        return X
    
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable):
        self.variable = variable

    def fit(self,X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        self.fill_values_= X[self.variable].mode()[0]
        return self

    def transform(self,X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill_values_)
        return X

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self,target_col: str):
        self.target_col=target_col

    def fit(self,X,y=None):
        X = X.copy()
        q1 = X.describe()[self.target_col].loc['25%']
        q3 = X.describe()[self.target_col].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self

    def transform(self,X):
        X_imputed = X.copy()
        for i in X[self.target_col].index:
          if X.loc[i,self.target_col] > self.upper_bound:
              X_imputed.loc[i,self.target_col] = self.upper_bound
          if X.loc[i,self.target_col] < self.lower_bound:
              X_imputed.loc[i,self.target_col] = self.lower_bound
        return X_imputed

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self,target_col:str):
        self.target_col = target_col
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self,X,y=None):
        X = X.copy()
        self.encoder.fit(X[[self.target_col]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out([self.target_col])
        return self

    def transform(self,X):
        # X_encoded = pd.get_dummies(X[self.target_col], prefix=self.target_col, drop_first=True)
        # return pd.concat([X, X_encoded], axis=1).drop(self.target_col, axis=1)
        X = X.copy()
        encoded_weekdays = self.encoder.transform(X[[self.target_col]])
        X[self.encoded_features_names] = encoded_weekdays
        X.drop(self.target_col, axis=1, inplace=True)  
        return X