# base_model.py (Güncellenmiş Versiyon)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class BaseTaxModel:
    def __init__(self, tax_name, frequency="monthly"):
        self.tax_name = tax_name
        self.frequency = frequency # "monthly" veya "quarterly"
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
        df_filtered.set_index('Period', inplace=True)
        
        # --- SİZİN BELİRTTİĞİNİZ DEĞİŞİM ORANI ALGORİTMASI ---
        if self.frequency == "monthly":
            # Aylık veri için: n / (n-12) -> periods=12
            # pandas pct_change(12) tam olarak (n - n-12) / n-12 * 100 hesaplar
            df_pct = df_filtered.pct_change(periods=12) * 100
        elif self.frequency == "quarterly":
            # Çeyreklik veri için: n / (n-4) -> periods=4
            df_pct = df_filtered.pct_change(periods=4) * 100
        else:
            # Varsayılan düz zincirleme değişim
            df_pct = df_filtered.pct_change() * 100
            
        # Değişim oranından dolayı ilk 4 veya 12 satır NaN (boş) olacaktır, onları temizliyoruz
        df_pct.dropna(inplace=True)
        
        # Sonsuz veya hatalı değerleri (0'a bölünme vb.) düzeltme
        df_pct.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Kronolojik Train / Test Ayrımı
        split_idx = int(len(df_pct) * train_ratio)
        train_df = df_pct.iloc[:split_idx]
        test_df = df_pct.iloc[split_idx:]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Verileri Ölçeklendirme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, y_train, X_test_scaled, y_test, test_df

    def build_model(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        if self.model is None:
            self.build_model(X_train.shape[1])
            
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=len(X_train), # Full-batch
            verbose=0
        )
        return history

    def predict(self, X_scaled):
        return self.model.predict(X_scaled).flatten()

    def expert_rules(self, raw_prediction, current_features):
        return raw_prediction