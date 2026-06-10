# tax_models.py
from base_model import BaseTaxModel

# tax_models.py içindeki PITModel sınıfı

class PITModel(BaseTaxModel):
    """ Gelir Vergisi Tahmin Modeli (Çeyreklik) """
    def __init__(self):
        super().__init__(tax_name="Gelir Vergisi (PIT)", frequency="quarterly")
        
    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı: Gelir vergisinin %75'ini oluşturan ücretler ile 
        %25'ini oluşturan menkul sermaye iratlarının (MSİ) indikatör bazlı 
        ağırlıklı ekonomik beklentisini hesaplar ve ham tahmini düzeltir.
        """
        # current_features içindeki değerler YoY % değişim oranlarıdır.
        
        # 1. Ücretler Grubu Beklenen Değişimi (%75 Ağırlık)
        # Maaş endeksi artışı ve çalışan hacmi artışının ortalaması
        wage_trend = (current_features['Salary_Inx'] + current_features['Employee_Vol']) / 2
        
        # 2. Menkul Sermaye İratları (MSİ) Grubu Beklenen Değişimi (%25 Ağırlık)
        # Mevduat hacmi artışı ve mevduat faiz oranı artışının ortalaması
        msi_trend = (current_features['Deposit_Vol'] + current_features['Deposit_Int_Rt']) / 2
        
        # 3. Toplam Ekonomik Gösterge Trendi (Ağırlıklı Ortalama)
        economic_expected_change = (wage_trend * 0.75) + (msi_trend * 0.25)
        
        # 4. Düzeltme Mekanizması: 
        # Yapay zekanın ham tahmini ile ekonomik beklenti arasında bir orta yol buluyoruz.
        # Örneğin: Modelin ham tahmini ile ekonomik beklentinin ortalamasını alarak 
        # sinir ağının aşırı (overfitting/underfitting) sapmalarını törpülüyoruz.
        adjusted_prediction = (raw_prediction + economic_expected_change) / 2
        
        # Alternatif olarak daha esnek bir süzgeç de koyabilirsiniz:
        # if abs(raw_prediction - economic_expected_change) > 15:
        #     # Eğer model ekonomik göstergelerden %15'ten fazla saptıysa, ekonomik trende doğru %30 yaklaştır:
        #     adjusted_prediction = (raw_prediction * 0.7) + (economic_expected_change * 0.3)
            
        return adjusted_prediction


# tax_models.py içindeki CITModel sınıfı

class CITModel(BaseTaxModel):
    """ Kurumlar Vergisi Tahmin Modeli (Çeyreklik) """
    def __init__(self):
        super().__init__(tax_name="Kurumlar Vergisi (CIT)", frequency="quarterly")
        
    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı: Kurumlar vergisinin %75'ini oluşturan Reel Sektör indikatörleri 
        ile %25'ini oluşturan Bankacılık Sektörü indikatörlerinin ağırlıklı 
        ekonomik trendini hesaplar ve yapay zeka modelinin ham tahminini dengeler.
        """
        # current_features içindeki değerler YoY % değişim oranlarıdır.
        
        # 1. Reel Sektör Grubu Beklenen Trendi (%75 Ağırlık)
        # TÜFE (TUFE_Inx), İmalat Sanayi (Man_Prod_Inx), Ticaret Hacmi (Trade_Vol_Inx) 
        # ve Reel Sektör Vergi Oranı (Reel_Sector_Tax_Rt) değişimlerinin ortalaması
        reel_sector_trend = (
            current_features['CPI_Inx'] + 
            current_features['Man_Prod_Inx'] + 
            current_features['Trade_Vol_Inx'] + 
            current_features['Reel_Sector_Tax_Rt']
        ) / 4
        
        # 2. Bankacılık Sektörü Grubu Beklenen Trendi (%25 Ağırlık)
        # Banka Faiz Marjı (Banking_Int_Rate_Margin) ve Sektörel Vergi Oranı (Banking_Sector_Tax_Rt)
        banking_sector_trend = (
            current_features['Banking_Int_Rate_Margin'] + 
            current_features['Banking_Sector_Tax_Rt']
        ) / 2
        
        # 3. Toplam Kurumsal Ekonomik Beklenti Trendi (Ağırlıklı Ortalama)
        corporate_expected_change = (reel_sector_trend * 0.75) + (banking_sector_trend * 0.25)
        
        # 4. Düzeltme / Filtreleme Mekanizması:
        # Derin öğrenme modelinin ürettiği ham tahmin ile bu yapısal ekonomik 
        # beklenti trendinin dengeli bir kombinasyonunu alıyoruz.
        adjusted_prediction = (raw_prediction + corporate_expected_change) / 2
        
        # Gelişmiş Senaryo Koruması (Opsiyonel):
        # Eğer bankacılık faiz marjlarında olağanüstü bir şok veya vergi oranlarında 
        # mevzuatsal sert bir değişim varsa modelin rasyonel sınırlarda kalmasını garanti eder.
        if abs(raw_prediction - corporate_expected_change) > 20:
            # Model ekonomik gerçeklikten %20'den fazla saptıysa, ekonomik trend ağırlığını artır:
            adjusted_prediction = (raw_prediction * 0.6) + (corporate_expected_change * 0.4)
            
        return adjusted_prediction

class ImportVATModel(BaseTaxModel):
    """ İthalatta Alınan KDV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="İthalat KDV", frequency="monthly")

class DomesticVATModel(BaseTaxModel):
    """ Dahili KDV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Dahili KDV", frequency="monthly")

class PetroleumSCTModel(BaseTaxModel):
    """ Petrol ve Doğalgaz ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Petrol ÖTV", frequency="monthly")

class TobaccoSCTModel(BaseTaxModel):
    """ Tütün ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Tütün ÖTV", frequency="monthly")

class MotorVehiclesSCTModel(BaseTaxModel):
    """ Motorlu Taşıtlar ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Motorlu Taşıtlar ÖTV", frequency="monthly")