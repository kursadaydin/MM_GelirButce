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
    def __init__(self): super().__init__(tax_name="Dahili KDV", frequency="monthly")
    def expert_rules(self, raw_prediction, current_features): return raw_prediction

class PetroleumSCTModel(BaseTaxModel):
    def __init__(self): super().__init__(tax_name="Petrol ÖTV", frequency="monthly")
    def expert_rules(self, raw_prediction, current_features): return raw_prediction

class TobaccoSCTModel(BaseTaxModel):
    def __init__(self): super().__init__(tax_name="Tütün ÖTV", frequency="monthly")
    def expert_rules(self, raw_prediction, current_features): return raw_prediction

class MotorVehiclesSCTModel(BaseTaxModel):
    def __init__(self): super().__init__(tax_name="Motorlu Taşıtlar ÖTV", frequency="monthly")
    def expert_rules(self, raw_prediction, current_features): return raw_prediction