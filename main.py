# main.py
import os
import sys

# Windows Sistem Seviyesinde venv NVIDIA DLL Enjeksiyonu
venv_base = sys.prefix
nvidia_bin_path = os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "bin")
if os.path.exists(nvidia_bin_path):
    os.add_dll_directory(nvidia_bin_path)
    for folder in os.listdir(os.path.join(venv_base, "Lib", "site-packages", "nvidia")):
        bin_dir = os.path.join(venv_base, "Lib", "site-packages", "nvidia", folder, "bin")
        if os.path.exists(bin_dir):
            os.add_dll_directory(bin_dir)

# Uyarılardan arındırılmış temiz terminal modu
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from src import tax_models

# Excel dosyasının adı ve içindeki SAYFA (Sheet) isimleri konfigürasyonu
EXCEL_FILE_NAME = "data_ml_tax.xlsx"

TAX_CONFIGS = {
    "PIT": {
        "sheet_name": "PIT",  # Excel'deki sayfa adı
        "model_class": tax_models.PITModel,
        "target": "Tax_PIT",
        "features": ["Salary_Inx", "Employee_Vol", "Deposit_Vol", "Deposit_Int_Rt"]
    },
    "CIT": {
        "sheet_name": "CIT",
        "model_class": tax_models.CITModel,
        "target": "Tax_CIT",
        "features": ["CPI_Inx", "Man_Prod_Inx", "Trade_Vol_Inx", "Reel_Sector_Tax_Rt", "Banking_Int_Rate_Margin", "Banking_Sector_Tax_Rt"]
    },
    "Import_VAT": {
        "sheet_name": "Import VAT",
        "model_class": tax_models.ImportVATModel,
        "target": "Tax_Import_VAT",
        "features": ["Import_ExcGOLD_USD_Vol", "Ave_Dolar_Exc_Rt", "Import_VAT_Effective_Tax_Rt"]
    },
    "Domestic_VAT": {
        "sheet_name": "Domestic VAT",
        "model_class": tax_models.DomesticVATModel,
        "target": "Tax_Domestic_VAT",
        "features": ["CPI_Inx", "Trade_Vol_Inx", "CreditCart_Usage_Vol", "Tax_Total_SCT"]
    },
    "Petroleum_SCT": {
        "sheet_name": "Petroleum SCT",
        "model_class": tax_models.PetroleumSCTModel,
        "target": "Tax_Petrelum_SCT",
        "features": ["Cons_Gasoline", "Cons_Diesel", "Fix_Tax_Gasoline", "Fix_Tax_Diesel"]
    },
    "Tobacco_SCT": {
        "sheet_name": "Tobacco SCT",
        "model_class": tax_models.TobaccoSCTModel,
        "target": "Tax_Tobacco_SCT",
        "features": ["Cons_Tobacco", "Advolorem_Tax_Tobacco"]
    },
    "Motor_Vehicles_SCT": {
        "sheet_name": "Motor Vehicles SCT",
        "model_class": tax_models.MotorVehiclesSCTModel,
        "target": "Tax_Motor_Vehicle_SCT",
        "features": ["Motor_Vehicle_Sales_Vol", "Motor_Vehicle_Average_Tax_Base", "Motor_Vehicle_Effective_Tax_Rt", "Electrical_Vehicle_Sales_Vol"]
    }
}

def main():
    print("="*60)
    print("DEEP LEARNING VERGİ TAHMİN SİSTEMİ BAŞLATILIYOR")
    print("="*60)
    
    # data/data_ml_tax.xlsx yolunu oluştur
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.normpath(os.path.join(base_dir, "data", EXCEL_FILE_NAME))
    
    if not os.path.exists(excel_path):
        print(f"❌ Hata: Ana veri dosyası bulunamadı! Aranan Yol: {excel_path}")
        return

    # Excel dosyasını bir kez hafızaya yükleyip içindeki sayfaları kontrol edelim
    try:
        excel_file = pd.ExcelFile(excel_path)
    except Exception as e:
        print(f"❌ Excel dosyası okunurken hata oluştu: {e}")
        return
    
    for tax_key, config in TAX_CONFIGS.items():
        sheet = config["sheet_name"]
        
        # Eğer Excel içinde bu isimde bir sayfa yoksa atla
        if sheet not in excel_file.sheet_names:
            print(f"⚠️ Uyarı: Excel içinde '{sheet}' isimli bir sayfa bulunamadı. Atlanıyor...")
            continue
            
        # Sayfayı oku
        df = excel_file.parse(sheet_name=sheet)
        
        model_instance = config["model_class"]()
        print(f"\n▶️ {model_instance.tax_name} Modeli Eğitiliyor (Sayfa: {sheet})...")
        
        # Pipeline çalıştırma
        X_train, y_train, X_test, y_test, test_df = model_instance.prepare_data(
            df, target_col=config["target"], feature_cols=config["features"]
        )
        
        model_instance.train(X_train, y_train, X_test, y_test, epochs=200)
        raw_preds = model_instance.predict(X_test)
        
        # Uzman Kuralları Uygulama
        final_preds = []
        for i, raw_pred in enumerate(raw_preds):
            # Doğrudan satırın tamamını (tüm _YoY sütunlarını içerecek şekilde) gönderiyoruz
            current_row_features = test_df.iloc[i] 
            adjusted_pred = model_instance.expert_rules(raw_pred, current_row_features)
            final_preds.append(adjusted_pred)
            
        final_preds = np.array(final_preds)
        mae = np.mean(np.abs(final_preds - y_test))
        print(f"✅ {model_instance.tax_name} için Test Seti MAE Skoru: {mae:.4f}")

if __name__ == "__main__":
    main()