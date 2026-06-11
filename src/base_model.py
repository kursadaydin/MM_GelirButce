# src/base_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class BaseTaxModel:
    def __init__(self, tax_name, frequency="monthly"):
        self.tax_name = tax_name
        self.frequency = frequency  # "monthly" veya "quarterly"
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = ""

    def prepare_data(self, df, target_col, feature_cols, train_ratio=0.8):
        """
        Veriyi frekansına göre geçen yılın aynı dönemine (YoY) kıyaslayarak
        yüzdesel değişime çevirir, Train/Test olarak ayırır.
        """
        self.target_column = target_col
        self.feature_columns = feature_cols
        
        df_filtered = df[['Period'] + feature_cols + [target_col]].copy()
        
        # --- TÜRKÇE ÇEYREKLİK FORMATI (2018 Ç1 -> 2018Q1) DÜZELTME KATMANI ---
        df_filtered['Period'] = df_filtered['Period'].astype(str).str.strip()
        
        if self.frequency == "quarterly":
            df_filtered['Period'] = df_filtered['Period'].str.replace('Ç', 'Q', regex=False).str.replace(' ', '', regex=False)
            df_filtered['Period'] = pd.to_datetime(df_filtered['Period'])
        else:
            df_filtered['Period'] = pd.to_datetime(df_filtered['Period'], errors='coerce')
            
        df_filtered.set_index('Period', inplace=True)
        df_filtered.sort_index(inplace=True)
        
        # Değişim oranı hesaplama (n / n-12 veya n / n-4)
        if self.frequency == "monthly":
            df_pct = df_filtered.pct_change(periods=12) * 100
        else:
            df_pct = df_filtered.pct_change(periods=4) * 100
            
        df_pct.dropna(inplace=True)
        
        # --- SÜTUN İSİMLERİNE _YoY SON EKİ EKLEME (KeyError Çözümü) ---
        # Hem feature hem de target sütunlarının sonuna _YoY ekliyoruz
        rename_dict = {col: f"{col}_YoY" for col in feature_cols + [target_col]}
        df_pct.rename(columns=rename_dict, inplace=True)
        
        yoy_features = [f"{col}_YoY" for col in feature_cols]
        yoy_target = f"{target_col}_YoY"
        
        # Kronolojik Ayrım (%80 Train, %20 Test)
        split_idx = int(len(df_pct) * train_ratio)
        train_df = df_pct.iloc[:split_idx]
        test_df = df_pct.iloc[split_idx:]
        
        X_train = train_df[yoy_features].values
        y_train = train_df[yoy_target].values
        X_test = test_df[yoy_features].values
        y_test = test_df[yoy_target].values
        
        # Verileri Ölçeklendirme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, y_train, X_test_scaled, y_test, test_df

    def build_model(self, input_dim):
        # Keras 3.x modern Sequential mimarisi (Input katmanı ile)
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        if self.model is None:
            self.build_model(X_train.shape[1])
            
        # Full-batch training
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=len(X_train),
            verbose=0
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def expert_rules(self, raw_prediction, current_features):
        return raw_prediction