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
EXCEL_FILE_NAME = "data_ml_tax_v2.xlsx"

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
        "features": ["Average_Price_Pack_Cigarette","Cons_Tobacco", "Advolorem_Tax_Tobacco","Specific_Tax_Tobacco","Min_Specific_Tax_Tobacco"]
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
        
        # Modeli eğit ve ham tahminleri al
        model_instance.train(X_train, y_train, X_test, y_test, epochs=200)
        raw_preds = model_instance.predict(X_test).flatten() # Boyut uyuşmazlığını önlemek için düzleştirildi
        
        # =====================================================================
        # DÜZELTİLEN KISIM: UZMAN KURALLARININ ÇALIŞTIRILMASI VE LİSTE YAPISI
        # =====================================================================
        final_preds_list = []
        for i, raw_pred in enumerate(raw_preds):
            current_row_features = test_df.iloc[i]
            # Her bir tahmin için modele özgü yazılmış expert_rules tetikleniyor
            adjusted_pred = model_instance.expert_rules(raw_pred, current_row_features)
            final_preds_list.append(adjusted_pred)
            
        final_preds = np.array(final_preds_list)
        
        # Gerçek hedef değişkenin YoY değerleri (Zaten tek boyutlu dizidir)
        y_test_yoy = y_test.flatten() if hasattr(y_test, "flatten") else np.array(y_test)
        
        # MAE ve MSE Hesaplama
        mae = np.mean(np.abs(y_test_yoy - final_preds))
        mse = np.mean((y_test_yoy - final_preds) ** 2)
        
        print(f"📊 {model_instance.tax_name} Modeli Performansı:")
        print(f"   -> Ortalama Mutlak Hata (MAE): {mae:.4f} %")
        print(f"   -> Ortalama Kare Hata (MSE): {mse:.4f} %")
        
        # =====================================================================
        # 🎯 DÖNEM BAZLI TAHMİN VE GERÇEK DEĞER PRİNTLERİ
        # =====================================================================
        print(f"\n🔮 {model_instance.tax_name} - Test Dönemi Tahmin Sonuçları (YoY %):")
        print("-" * 85)
        print(f"{'Dönem':<12} | {'Gerçek YoY (%)':<16} | {'Model Ham Tahmin (%)':<22} | {'Uzman Ayarlı Tahmin (%)':<24}")
        print("-" * 85)
        
        # Güvenli tarih erişimi için test_df indeksini kontrol et/oluştur
        if 'Period' in test_df.columns:
            periods = pd.to_datetime(test_df['Period'])
        else:
            periods = pd.to_datetime(test_df.index)
        
        # Test veri kümesindeki her bir satır (dönem) için sonuçları yazdır
        for idx in range(len(test_df)):
            current_date = periods[idx]
            
            # Çeyreklik veya aylık frekansa göre ekranda güzel görünmesini sağlıyoruz
            if model_instance.frequency == "quarterly":
                quarter = (current_date.month - 1) // 3 + 1
                period_str = f"{current_date.year}Q{quarter}"
            else:
                period_str = current_date.strftime('%Y-%m')
            
            actual_val = y_test_yoy[idx]
            raw_pred_val = raw_preds[idx]
            final_pred_val = final_preds[idx]
            
            print(f"{period_str:<12} | {actual_val:<16.2f} | {raw_pred_val:<22.2f} | {final_pred_val:<24.2f}")
            
        print("-" * 85)
        print("=" * 60) # Modeller arası görsel ayrım

if __name__ == "__main__":
    main()