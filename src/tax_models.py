# tax_models.py
from base_model import BaseTaxModel

# tax_models.py içindeki PITModel sınıfı

class PITModel(BaseTaxModel):
    """ Gelir Vergisi Tahmin Modeli (Çeyreklik) """
    def __init__(self):
        super().__init__(tax_name="Gelir Vergisi (PIT)", frequency="quarterly")
        
    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı: Gelir vergisinin indikatörler ile ölçülebilen %74'lük kısmını
        (Ücret: %54, MSİ: %20) ve indikatör kısıtından dolayı ölçülemeyen %26'lık 
        kör noktayı bütçe paylarına göre ağırlıklandırarak ham tahmini rasyonelleştirir.
        """
        # current_features içindeki değerler YoY % değişim oranlarıdır.
        
        # 1. Ücretler Grubu Beklenen Trendi (Toplam bütçedeki payı: %54)
        # Maaş endeksi ve çalışan hacmi ortalaması
        wage_trend = (current_features['Salary_Inx'] + current_features['Employee_Vol']) / 2
        
        # 2. Menkul Sermaye İratları (MSİ) Grubu Beklenen Trendi (Toplam bütçedeki payı: %20)
        # Mevduat hacmi ve mevduat faiz oranı ortalaması
        msi_trend = (current_features['Deposit_Vol'] + current_features['Deposit_Int_Rt']) / 2
        
        # 3. İndikatörler ile Ölçülebilen Güvenli Alan Trendi (%74 Ağırlık)
        measured_expected_change = (wage_trend * 0.54) + (msi_trend * 0.20)
        
        # 4. Ölçülemeyen/Kör Nokta Grubu Trendi (%26 Ağırlık)
        # Veri setinde doğrudan ölçülemeyen bu pay için en rasyonel yaklaşım;
        # yapay zekanın ürettiği ham tahmin trendini baz almak ya da ölçülebilen trende eşitlemektir.
        # Burada yapay zekanın ham tahmininin %26'lık temsil gücünü koruyoruz:
        unmeasured_expected_change = raw_prediction * 0.26
        
        # 5. Toplam Ekonomik Yapısal Beklenti
        # Ölçülebilen indikatör trendi ile ölçülemeyen kısmın ağırlıklı birleşimi
        total_economic_expected = measured_expected_change + unmeasured_expected_change
        
        # 6. Düzeltme Mekanizması:
        # Yapay sinir ağının ham tahmini ile bütçe kırılımlı ekonomik beklentiyi dengeliyoruz.
        # Böylece model veri eksikliğinden dolayı %26'lık kısmı yanlış yorumlarsa süzgeç devreye girer.
        adjusted_prediction = (raw_prediction + total_economic_expected) / 2
        
        # Aşırı sapma filtresi (Model bütçe mantığından %15'ten fazla uzaklaşırsa):
        if abs(raw_prediction - total_economic_expected) > 15:
            adjusted_prediction = (raw_prediction * 0.6) + (total_economic_expected * 0.4)
            
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

    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı Havuzu: İlerleyen aşamalarda gümrük tarifeleri, 
        kur şokları ve ithalat kısıtlamaları süzgeçleri eklenecektir.
        """
        adjusted_prediction = raw_prediction
        # TODO: İthalat KDV uzman kuralları buraya yazılacak.
        return adjusted_prediction


class DomesticVATModel(BaseTaxModel):
    """ Dahili KDV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Dahili KDV", frequency="monthly")

    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı Havuzu: İlerleyen aşamalarda harcama limitleri, 
        kredi kartı taksit sınırlamaları ve temel KDV oran değişiklikleri eklenecektir.
        """
        adjusted_prediction = raw_prediction
        # TODO: Dahili KDV uzman kuralları buraya yazılacak.
        return adjusted_prediction


class PetroleumSCTModel(BaseTaxModel):
    """ Petrol ve Doğalgaz ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Petrol ÖTV", frequency="monthly")

    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı Havuzu: İlerleyen aşamalarda maktu vergi güncellemeleri 
        ve Eşel Mobil Sistemi yasal sınırları entegre edilecektir.
        """
        adjusted_prediction = raw_prediction
        # TODO: Petrol ÖTV uzman kuralları buraya yazılacak.
        return adjusted_prediction


class TobaccoSCTModel(BaseTaxModel):
    """ Tütün ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Tütün ÖTV", frequency="monthly")

    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı Havuzu: İlerleyen aşamalarda nispi/maktu vergi eşikleri 
        ve asgari maktu tutar güncellemeleri eklenecektir.
        """
        adjusted_prediction = raw_prediction
        # TODO: Tütün ÖTV uzman kuralları buraya yazılacak.
        return adjusted_prediction


class MotorVehiclesSCTModel(BaseTaxModel):
    """ Motorlu Taşıtlar ÖTV (Aylık) """
    def __init__(self):
        super().__init__(tax_name="Motorlu Taşıtlar ÖTV", frequency="monthly")

    def expert_rules(self, raw_prediction, current_features):
        """
        Uzman Kuralı Havuzu: İlerleyen aşamalarda matrah dilimleri şokları, 
        GSR II güvenlik regülasyonları ve elektrikli araç (EV) pazar payı geçişleri eklenecektir.
        """
        adjusted_prediction = raw_prediction
        # TODO: Motorlu Taşıtlar ÖTV uzman kuralları buraya yazılacak.
        return adjusted_prediction