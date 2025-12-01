"""
Transformadores personalizados para el pipeline de datos del cliente.
Este módulo debe estar disponible tanto al crear el pipeline como al cargarlo.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Genera la columna 'DaysSinceLast' basada en la diferencia de fechas.
    Debe ejecutarse ANTES de DropColumns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Creamos una copia para no afectar el dataframe original fuera del pipeline
        X = X.copy()
        
        # Verificamos que las columnas existan antes de operar
        if 'TransactionDate' in X.columns and 'PreviousTransactionDate' in X.columns:
            # Asegurar tipo datetime
            X['TransactionDate'] = pd.to_datetime(X['TransactionDate'])
            X['PreviousTransactionDate'] = pd.to_datetime(X['PreviousTransactionDate'])
            
            # Calcular diferencia
            # Nota: Según tu lógica es Previous - Transaction
            X['TimeSinceLastTransaction'] = X['PreviousTransactionDate'] - X['TransactionDate']
            
            # Convertir a días (segundos totales / 86400)
            X['DaysSinceLast'] = X['TimeSinceLastTransaction'].dt.total_seconds() / 86400
            
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Elimina columnas innecesarias del dataset de clientes.
    """
    def __init__(self):
        self.columns = [
            'CustomerID', 'Id Complain', 'Id Interaction', 'date_received', 
            'Survey date', 'Twitter', 'NPS', 'product', 'sub_product', 
            'issue', 'sub_issue', 'Gender', 'TransactionID', 'AccountID', 
            'DeviceID', 'IP Address', 'MerchantID', "TransactionDate", 
            "PreviousTransactionDate", 'TimeSinceLastTransaction'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor dinámico que:
    - Escala variables numéricas con MinMaxScaler
    - Codifica variables categóricas con OneHotEncoder
    """
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.num_scaler = MinMaxScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False)
        self.cat_feature_names = []

    def fit(self, X, y=None):
        # Identificar columnas numéricas y categóricas
        self.num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Ajustar scalers
        if self.num_cols:
            self.num_scaler.fit(X[self.num_cols])
        
        if self.cat_cols:
            self.cat_encoder.fit(X[self.cat_cols])
            self.cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols)
        
        return self

    def transform(self, X):
        # Transformar numéricas
        num_part = self.num_scaler.transform(X[self.num_cols]) if self.num_cols else np.empty((len(X), 0))
        
        # Transformar categóricas
        cat_part = self.cat_encoder.transform(X[self.cat_cols]) if self.cat_cols else np.empty((len(X), 0))
        
        # Combinar
        data = np.hstack([num_part, cat_part])
        columns = self.num_cols + list(self.cat_feature_names)
        
        return pd.DataFrame(data, columns=columns, index=X.index)