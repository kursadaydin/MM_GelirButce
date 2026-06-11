# src/tax_models.py
import numpy as np
from .base_model import BaseTaxModel

class PITModel(BaseTaxModel):
    """ Gelir Vergisi Tahmin Modeli (Çeyreklik) """
    def __init__(self):
        super().__init__(tax_name="Gelir Vergisi (PIT)", frequency="quarterly")
        
    def expert_rules(self, raw_prediction, current_features):
        # 1. Ücretler Grubu Trendi (%54)
        wage_trend = (current_features['Salary_Inx_YoY'] + current_features['Employee_Vol_YoY']) / 2
        
        # 2. Menkul Sermaye İratları (MSİ) Grubu Trendi (%20)
        msi_trend = (current_features['Deposit_Vol_YoY'] + current_features['Deposit_Int_Rt_YoY']) / 2
        
        # 3. İndikatörler ile Ölçülebilen Güvenli Alan Trendi (%74 Ağırlık)
        measured_expected_change = (wage_trend * 0.54) + (msi_trend * 0.20)
        
        # 4. Ölçülemeyen/Kör Nokta Grubu Trendi (%26 Ağırlık)
        unmeasured_expected_change = raw_prediction * 0.26
        
        # 5. Toplam Ekonomik Yapısal Beklenti
        total_economic_expected = measured_expected_change + unmeasured_expected_change
        
        # 6. Düzeltme Mekanizması
        adjusted_prediction = (raw_prediction + total_economic_expected) / 2
        
        if abs(raw_prediction - total_economic_expected) > 15:
            adjusted_prediction = (raw_prediction * 0.6) + (total_economic_expected * 0.4)
            
        return adjusted_prediction

class CITModel(BaseTaxModel):
    """ Kurumlar Vergisi Tahmin Modeli (Çeyreklik) """
    def __init__(self):
        super().__init__(tax_name="Kurumlar Vergisi (CIT)", frequency="quarterly")
        
    def expert_rules(self, raw_prediction, current_features):
        # %75 Reel Sektör, %25 Finans Sektörü bütçe dengesi
        real_sector_trend = (current_features['CPI_Inx_YoY'] + current_features['Man_Prod_Inx_YoY'] + current_features['Trade_Vol_Inx_YoY']) / 3
        banking_sector_trend = current_features['Banking_Int_Rate_Margin_YoY']
        
        economic_expected = (real_sector_trend * 0.75) + (banking_sector_trend * 0.25)
        adjusted_prediction = (raw_prediction + economic_expected) / 2
        return adjusted_prediction

class ImportVATModel(BaseTaxModel):
    def __init__(self): super().__init__(tax_name="İthalat KDV", frequency="monthly")
    def expert_rules(self, raw_prediction, current_features): return raw_prediction

class DomesticVATModel(BaseTaxModel):
    """ Dahili KDV Tahmin Modeli (Aylık) """
    def __init__(self): 
        super().__init__(tax_name="Dahili KDV", frequency="monthly")
        
    def prepare_data(self, df, target_col, feature_cols):
        # 1. Orijinal df'i bozmamak için derin kopyalama yapıyoruz
        df_clean = df.copy()
        
        # 2. Excel kolon adlarında sağda solda gizli boşluklar varsa temizliyoruz (Güvenlik Önlemi)
        df_clean.columns = df_clean.columns.str.strip()
        
        # 3. Eğer konfigürasyondan gelen target_col ismi Excel'de tam eşleşmiyorsa dinamik yakalıyoruz
        # Örn: Excel'de "Gross_Tax_Domestic_VAT " veya küçük harfle "gross_tax_domestic_vat" olabilir.
        actual_target_col = target_col
        for col in df_clean.columns:
            if col.lower() == target_col.lower():
                actual_target_col = col
                break

        # 4. Matematiksel Arındırma İşlemi
        # 'Additional_Revenue' kolonu mevcutsa arındırmayı gerçek kolon üzerinden yapıyoruz
        if 'Additional_Revenue' in df_clean.columns:
            df_clean[actual_target_col] = df_clean[actual_target_col] - df_clean['Additional_Revenue']
            
        # 5. BaseTaxModel'e güncellenmiş dinamik hedef kolon adıyla gönderiyoruz
        return super().prepare_data(df_clean, target_col=actual_target_col, feature_cols=feature_cols)

    def expert_rules(self, raw_prediction, current_features):
        consumption_trend = (current_features['CPI_Inx_YoY'] + current_features['Trade_Vol_Inx_YoY'] + current_features['CreditCart_Usage_Vol_YoY']) / 3
        sct_effect = current_features['Tax_Total_SCT_YoY']
        organic_expected = (consumption_trend * 0.70) + (sct_effect * 0.30)
        adjusted_prediction = (raw_prediction * 0.70) + (organic_expected * 0.30)
        
        additional_revenue_effect = current_features.get('Additional_Revenue_YoY', 0)
        if additional_revenue_effect > 0:
            adjusted_prediction = adjusted_prediction + (additional_revenue_effect * 0.20)
            
        return adjusted_prediction


class PetroleumSCTModel(BaseTaxModel):
    """ Petrol ÖTV Tahmin Modeli (Aylık) """
    def __init__(self): 
        super().__init__(tax_name="Petrol ÖTV", frequency="monthly")
        
    def prepare_data(self, df, target_col, feature_cols):
        # 1. Orijinal df'i bozmamak için derin kopyalama yapıyoruz
        df_clean = df.copy()
        
        # 2. Excel kolon adlarındaki olası boşlukları temizliyoruz
        df_clean.columns = df_clean.columns.str.strip()
        
        # 3. Küçük/Büyük harf duyarlılığı kontrolü
        actual_target_col = target_col
        for col in df_clean.columns:
            if col.lower() == target_col.lower():
                actual_target_col = col
                break
        
        # 4. Matematiksel Rekonstrüksiyon: Kaybı hedefe geri ekliyoruz
        if 'Sliding_Scale_Mech' in df_clean.columns:
            df_clean[actual_target_col] = df_clean[actual_target_col] + df_clean['Sliding_Scale_Mech']
            
        return super().prepare_data(df_clean, target_col=actual_target_col, feature_cols=feature_cols)
        
    def expert_rules(self, raw_prediction, current_features):
        consumption_trend = (current_features['Cons_Gasoline_YoY'] + current_features['Cons_Diesel_YoY']) / 2
        fixed_tax_trend = (current_features['Fix_Tax_Gasoline_YoY'] + current_features['Fix_Tax_Diesel_YoY']) / 2
        organic_expected_trend = (consumption_trend * 0.40) + (fixed_tax_trend * 0.60)
        adjusted_prediction = (raw_prediction * 0.70) + (organic_expected_trend * 0.30)
        
        sliding_scale_effect = current_features.get('Sliding_Scale_Mech_YoY', 0)
        if sliding_scale_effect > 0:
            adjusted_prediction = adjusted_prediction - (sliding_scale_effect * 0.25)
            
        return adjusted_prediction


class PetroleumSCTModel(BaseTaxModel):
    """ Petrol ÖTV Tahmin Modeli (Aylık) """
    def __init__(self): 
        super().__init__(tax_name="Petrol ÖTV", frequency="monthly")
        
    def prepare_data(self, df, target_col, feature_cols):
        # Orijinal veriyi korumak için kopyalıyoruz
        df_clean = df.copy()
        
        # Matematiksel Rekonstrüksiyon:
        # Eğer eşel-mobil olmasaydı tahsil edilecek olan "Potansiyel Organik ÖTV"yi hedef değişken yapıyoruz.
        # Not: Sliding_Scale_Mech veri setinde feragat edilen pozitif bir tutar/matrah olarak yer alıyorsa topluyoruz.
        if 'Sliding_Scale_Mech' in df_clean.columns:
            df_clean[target_col] = df_clean[target_col] + df_clean['Sliding_Scale_Mech']
            
        return super().prepare_data(df_clean, target_col, feature_cols)
        
    def expert_rules(self, raw_prediction, current_features):
        # 1. Organik Potansiyel Trend Beklentisi
        # Tüketim miktarı (Benzin ve Motorin) trendi
        consumption_trend = (current_features['Cons_Gasoline_YoY'] + current_features['Cons_Diesel_YoY']) / 2
        
        # Maktu vergi tutarlarındaki yasal artış trendi
        fixed_tax_trend = (current_features['Fix_Tax_Gasoline_YoY'] + current_features['Fix_Tax_Diesel_YoY']) / 2
        
        # Katsayılar artık tamamen organik yapıya odaklı (%40 Hacim, %60 Yasal Tarife)
        organic_expected_trend = (consumption_trend * 0.40) + (fixed_tax_trend * 0.60)
        
        # Model potansiyel temiz veriyi öğrendiği için organik tahmine güvenimizi artırıyoruz
        adjusted_prediction = (raw_prediction * 0.70) + (organic_expected_trend * 0.30)
        
        # 2. Eşel-Mobil (Sliding Scale) Kaybının Düzenleme Olarak Düşülmesi:
        # Model bize potansiyel olması gereken ÖTV trendini söyledi. 
        # Şimdi bütçedeki net nakit gerçekleşmesini bulmak için feragat edilen tutarın trend etkisini düşüyoruz.
        sliding_scale_effect = current_features.get('Sliding_Scale_Mech_YoY', 0)
        
        if sliding_scale_effect > 0:
            # Sistem feragat yönlü çalışıyorsa (yani veri setindeki feragat tutarı YoY olarak büyüyorsa),
            # bu durum net tahsilat trendini negatif etkileyecektir.
            adjusted_prediction = adjusted_prediction - (sliding_scale_effect * 0.25)
            
        return adjusted_prediction

class TobaccoSCTModel(BaseTaxModel):
    """ Tütün ÖTV Tahmin Modeli (Aylık) """
    def __init__(self): 
        super().__init__(tax_name="Tütün ÖTV", frequency="monthly")
        
    def expert_rules(self, raw_prediction, current_features):
        # Tütün tüketim hacmi trendi
        volume_trend = current_features['Cons_Tobacco_YoY']
        
        # Ortalama paket fiyatı trendi (Nisbi verginin matrahını oluşturur)
        price_trend = current_features['Average_Price_Pack_Cigarette_YoY']
        
        # Yasal vergi parametrelerinin değişim trendleri
        advalorem_trend = current_features['Advolorem_Tax_Tobacco_YoY']  # Nisbi oran değişimi
        specific_trend = current_features['Specific_Tax_Tobacco_YoY']    # Maktu vergi değişimi
        min_specific_trend = current_features['Min_Specific_Tax_Tobacco_YoY'] # Asgari maktu vergi değişimi
        
        # Tütün ÖTV Denklem Karşılığı: Max(Fiyat * Nisbi, Asgari Maktu) + Maktu
        # Bu karmaşık yapıyı trend bazında ağırlıklandırıyoruz:
        # Fiyat ve nisbi oran doğrudan ciro bazlı (advalorem) etki yaratır.
        advalorem_effect = price_trend + advalorem_trend
        
        # Eğer fiyat artışı düşük kalırsa 'Asgari Maktu' taban fiyatı devreye girer ve koruma sağlar
        # Bu nedenle asgari maktu ve maktu vergi trendleri modele güçlü birer çıpa (anchor) sağlar.
        policy_effect = (specific_trend * 0.4) + (min_specific_trend * 0.6)
        
        # Hacim çarpanı ile birleştirilmiş toplam yasal/ekonomik beklenti
        total_tax_trend = volume_trend + (advalorem_effect * 0.5) + (policy_effect * 0.5)
        
        # Derin öğrenme ham tahmini ile uzman kuralı trendini konsolide etme
        adjusted_prediction = (raw_prediction * 0.5) + (total_tax_trend * 0.5)
        return adjusted_prediction


class MotorVehiclesSCTModel(BaseTaxModel):
    """ Motorlu Taşıtlar ÖTV Tahmin Modeli (Aylık) """
    def __init__(self): 
        super().__init__(tax_name="Motorlu Taşıtlar ÖTV", frequency="monthly")
        
    def expert_rules(self, raw_prediction, current_features):
        # Otomotiv sektörü satış hacmi trendi
        sales_trend = current_features['Motor_Vehicle_Sales_Vol_YoY']
        
        # Matrah dilimleri ve araç fiyatlarındaki değişim trendi (Efektif vergi oranına yansır)
        effective_rate_trend = current_features['Motor_Vehicle_Effective_Tax_Rt_YoY']
        base_trend = current_features['Motor_Vehicle_Average_Tax_Base_YoY']
        
        # Elektrikli araç teşvikleri/satış trendi (Genelde daha düşük ÖTV oranına tabi olduklarından marjinal seyreltme etkisi yapabilir)
        ev_trend = current_features.get('Electrical_Vehicle_Sales_Vol_YoY', 0)
        
        # Organik ÖTV matrah ve oran beklentisi
        market_trend = sales_trend + base_trend + effective_rate_trend
        
        if ev_trend > sales_trend:
            # Eğer elektrikli araç satış hızı toplam pazardan yüksekse, efektif vergi oranındaki kırılmayı modellemek adına beklentiyi hafif törpüle
            market_trend = market_trend - (ev_trend * 0.05)
            
        adjusted_prediction = (raw_prediction * 0.7) + (market_trend * 0.3)
        return adjusted_prediction