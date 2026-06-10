# main.py
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src import tax_models


# .env dosyasındaki parametreleri yükle
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "./data")
EPOCHS = int(os.getenv("MAX_EPOCHS", 200))

# Veri dosyaları ve bunlara ait hedef/açıklayıcı değişken eşlemeleri
TAX_CONFIGS = {
    "PIT": {
        "file": "data_ml_tax.xlsx - PIT.csv",
        "model_class": tax_models.PITModel,
        "target": "Tax_PIT",
        "features": ["Salary_Inx", "Employee_Vol", "Deposit_Vol", "Deposit_Int_Rt"]
    },
    "CIT": {
        "file": "data_ml_tax.xlsx - CIT.csv",
        "model_class": tax_models.CITModel,
        "target": "Tax_CIT",
        "features": ["TUFE_Inx", "Man_Prod_Inx", "Trade_Vol_Inx", "Reel_Sector_Tax_Rt", "Banking_Int_Rate_Margin", "Banking_Sector_Tax_Rt"]
    },
    "Import_VAT": {
        "file": "data_ml_tax.xlsx - Import VAT.csv",
        "model_class": tax_models.ImportVATModel,
        "target": "Tax_Import_VAT",
        "features": ["Import_ExcGOLD_USD_Vol", "Ave_Dolar_Exc_Rt", "Import_VAT_Effective_Tax_Rt"]
    },
    "Domestic_VAT": {
        "file": "data_ml_tax.xlsx - Domestic VAT.csv",
        "model_class": tax_models.DomesticVATModel,
        "target": "Tax_Domestic_VAT",
        "features": ["CPI_Inx", "Trade_Vol_Inx", "CreditCart_Usage_Vol", "Tax_Total_SCT"]
    },
    "Petroleum_SCT": {
        "file": "data_ml_tax.xlsx - Petroleum SCT.csv",
        "model_class": tax_models.PetroleumSCTModel,
        "target": "Tax_Petrelum_SCT",
        "features": ["Cons_Gasoline", "Cons_Diesel", "Fix_Tax_Gasoline", "Fix_Tax_Diesel"]
    },
    "Tobacco_SCT": {
        "file": "data_ml_tax.xlsx - Tobacco SCT.csv",
        "model_class": tax_models.TobaccoSCTModel,
        "target": "Tax_Tobacco_SCT",
        "features": ["Cons_Tobacco", "Advolorem_Tax_Tobacco"]
    },
    "Motor_Vehicles_SCT": {
        "file": "data_ml_tax.xlsx - Motor Vehicles SCT.csv",
        "model_class": tax_models.MotorVehiclesSCTModel,
        "target": "Tax_Motor_Vehicle_SCT",
        "features": ["Motor_Vehicle_Sales_Vol", "Motor_Vehicle_Average_Tax_Base", "Motor_Vehicle_Effective_Tax_Rt", "Electrical_Vehicle_Sales_Vol"]
    }
}

def main():
    print("="*60)
    print("DEEP LEARNING VERGİ TAHMİN SİSTEMİ BAŞLATILIYOR")
    print("="*60)
    
    for tax_key, config in TAX_CONFIGS.items():
        file_path = os.path.join(DATA_DIR, config["file"])
        
        if not os.path.exists(file_path):
            print(f"⚠️ Hata: {config['file']} dosyası data klasöründe bulunamadı. Atlanıyor...")
            continue
            
        # CSV dosyasını oku
        df = pd.read_csv(file_path)
        
        # Modeli ilklendir
        model_instance = config["model_class"]()
        print(f"\n▶️ {model_instance.tax_name} Modeli Eğitiliyor...")
        
        # Veriyi hazırla (Yüzdesel Değişim + Train/Test Split + Scaling)
        X_train, y_train, X_test, y_test, test_df = model_instance.prepare_data(
            df, target_col=config["target"], feature_cols=config["features"]
        )
        
        # Modeli Full-batch yöntemiyle eğit
        model_instance.train(X_train, y_train, X_test, y_test, epochs=EPOCHS)
        
        # Ham Tahminleri al
        raw_preds = model_instance.predict(X_test)
        
        # Uzman Kurallarını (Prompt Fonksiyonunu) Uygula
        final_preds = []
        for i, raw_pred in enumerate(raw_preds):
            current_row_features = test_df[config["features"]].iloc[i]
            # Vergiye özel yazılan kurallar burada devreye giriyor
            adjusted_pred = model_instance.expert_rules(raw_pred, current_row_features)
            final_preds.append(adjusted_pred)
            
        final_preds = np.array(final_preds)
        
        # Model Başarı Metriği: Ortalama Mutlak Hata (MAE) hesaplama
        mae = np.mean(np.abs(final_preds - y_test))
        print(f"✅ {model_instance.tax_name} için Test Seti MAE Skoru: {mae:.4f}")

if __name__ == "__main__":
    main()