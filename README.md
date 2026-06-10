DEEP LEARNING İLE TÜRKİYE VERGİ KALEMLERİ TAHMİN PROJESİ
DOKÜMANTASYON VE KILAVUZU

Bu proje, Altuğ Aydemir ve Cem Çebi tarafından hazırlanan "Forecasting Budgetary Items in Türkiye Using Deep Learning" (CBRT Working Paper No: 25/09) makalesindeki derin öğrenme metodolojisini temel alarak Türkiye'deki temel vergi gelirlerini tahmin etmek amacıyla geliştirilmiştir.

================================================================================

1. # PROJE METODOLOJİSİ & MAKALE KRİTERLERİ

Makalede bütçe ve vergi tahminleri için Yapay Sinir Ağları (ANN) ve Derin Sinir Ağları (DNN) modelleri kullanılmıştır. Kod mimarisinde sadık kalınacak temel parametreler ve gerekçeleri şunlardır:

• Aktivasyon Fonksiyonu: Doğrusal olmayan karmaşık şokları ve yapısal kırılmaları yakalayabilmek için ReLU (f(x) = max(0, x)) fonksiyonu tercih edilmiştir. ReLU, pozitif yönlü ekonomik trendleri hızla modele aktarır ve derin ağlarda öğrenmeyi kilitleyen "kaybolan gradyan" problemini engeller.

• Optimizasyon Algoritması: Esnek, kararlı ve adaptif öğrenme adımları için Adam (Adaptive Moment Estimation) algoritması kullanılmıştır. Adam, her bir ekonomik gösterge için ayrı bir öğrenme hızı belirler ve "momentum" etkisiyle hata oranını en güvenli şekilde küresel minimuma ulaştırır.

• Eğitim Yöntemi: Küçük veri gruplarının (mini-batch) yaratacağı kararsız dalgalanmaları ve zikzakları önlemek amacıyla Full-batch (Tam Yığın) yaklaşımı uygulanmıştır. Eğitim setindeki tüm satırlar modele tek bir büyük paket halinde verilir ve kararlı bir yakınsama (stable convergence) sağlanır.

• Performans Ölçütü: Modellerin tahmin başarısı ve kalitesi Ortalama Mutlak Hata (MAE - Mean Absolute Error) metriği ile değerlendirilecektir.

2. # ÇALIŞMA ALGORİTMASI VE VERİ İŞLEME HATTI (PIPELINE)
   ================================================================================

Proje altyapısı, ham verinin diskten okunmasından uzman kurallarının (prompt fonksiyonlarının) uygulanmasına kadar 7 ardışık adımdan oluşan dinamik bir veri işleme hattına (Pipeline) sahiptir:

[Ham Veri] -> [Dönem İndeksleme] -> [YoY % Değişim] -> [Kronolojik Bölümleme] -> [Standartlaştırma] -> [DNN Full-Batch Eğitim] -> [Uzman Kuralları Süzgeci] -> [MAE Skoru]

Adım 1: Veri Okuma ve Yapılandırma
Sistem, "main.py" dosyasındaki konfigürasyon haritasını tarayarak "./data/" klasöründeki kaynak verileri hafızaya yükler. "Period" sütunu zaman damgası indeksi olarak kurgulanarak verilerin kronolojik sırası güvenceye alınır.

Adım 2: Frekans Duyarlı Yıllıklandırılmış Değişim Oranı (YoY % Change)
Mali veriler, kurumsal takvimler ve yasal beyan dönemleri nedeniyle çok güçlü mevsimsel etkiler (seasonality) barındırır. Modelin bu dalgalanmalardan ötürü yanılmasını engellemek amacıyla, verilerin düz ardışık değişimi (n / n-1) yerine frekans yapısına göre "Geçen Yılın Aynı Dönemine Göre Yüzdesel Değişimi" hesaplanır:

• Aylık Veriler (KDV ve ÖTV kalemleri): Son 12 ayın aynı dönemine göre kıyaslama yapılır. Formül: [ (X_n - X_n-12) / X_n-12 ] _ 100
• Çeyreklik Veriler (PIT ve CIT kalemleri): Son 4 çeyreğin aynı dönemine göre kıyaslama yapılır. Formül: [ (X_n - X_n-4) / X_n-4 ] _ 100

Adım 3: Kronolojik Bölümleme (Train / Test Split)
Zaman serisi mantığı korunarak veriler rastgele karıştırılmadan bölünür. Verinin kronolojik olarak ilk %80'i Eğitim (Train), son %20'si ise Test (Out-of-sample) seti olarak ayrılır. Model, vergi gelirlerinin ekonomik göstergelerle bağını sadece eğitim setine bakarak öğrenir.

Adım 4: Standartlaştırma (Scaling)
Farklı ölçeklerdeki indikatörlerin (yüzdesel enflasyon oranları ile milyarlık işlem hacimleri gibi) model içindeki ağırlık dengesini bozmaması adına StandardScaler kullanılarak tüm girdiler ortalama = 0, standart sapma = 1 olacak şekilde normalize edilir.

Adım 5: Derin Sinir Ağı (DNN) Yapılanması ve Eğitim
Eğitim setindeki tüm satırlar, parçalara bölünmeden tek bir paket halinde (Full-batch) yapay sinir ağına beslenir. "Adam" optimizasyon algoritması, model her döngüyü (epoch) tamamladığında hata oranını azaltmak için katsayıları geriye yayılım (backpropagation) ile optimize eder.

Adım 6: Uzman Kuralları (Expert Rules / Prompt) Süzgeci
Model derin öğrenme tahminini ürettikten sonra, vergi uzmanının kurallarını barındıran post-processing katmanına girer. Her vergi kalemi için "tax_models.py" içinde yazılan kural fonksiyonları (Örn: "Enflasyon %60'ı aşarsa yurt içi KDV taban sınırını güncelle" veya "Mevduat faizleri yükseldiğinde stopaj etkisini modele ekle" gibi yasal kısıtlar) yapay zekanın ham tahminini rasyonelleştirir.

Adım 7: Performans Belirleme ve Hata Ölçümü
Uzman süzgecinden geçen nihai tahminler ile test setinde saklanan gerçek vergi gerçekleşmeleri karşılaştırılarak modelin hata skoru MAE cinsinden hesaplanır ve ekrana raporlanır.

# 3. PROJE KLASÖR MİMARİSİ

Projenin modüler, temiz ve nesne yönelimli (OOP) yapısı şu şekildedir:

vergi_tahmin_projesi/
│
├── data/ # Kaynak veriler (Excel'den üretilen CSV sayfaları)
│ ├── data_ml_tax.xlsx - PIT.csv
│ ├── data_ml_tax.xlsx - CIT.csv
│ ├── data_ml_tax.xlsx - Domestic VAT.csv
│ └── ... (Tüm vergi veri setleri)
│
├── src/ # Proje Kaynak Kodları (Modül Paketi)
│ ├── **init**.py # Python paket tanımlayıcısı
│ ├── base_model.py # Veri işleme hatlarını ve ana DNN sınıfını tutar
│ └── tax_models.py # Veriye özel uzman kurallarını (promptları) tutar
│
├── .env # Hiperparametre ve dizin ayarları
├── .gitignore # Git takip dışı bırakılacak dosyalar ve venv klasörü
├── main.py # Projeyi uçtan uca tetikleyen ve raporlayan ana script
├── requirements.txt # Bağımlı kütüphaneler listesi
└── README.md # Proje markdown dokümantasyonu

# 4. VERI SETI MIMARISI VE INDIKATÖRLER

Projede kullanılan girdi (feature) setleri ekonomik teoriye ve veri setinin güncel yapısına uygun olarak şu şekilde eşleştirilmiştir:

1. Gelir Vergisi (Tax_PIT) - Çeyreklik Veri (n / n-4)
   • Bağımlı Değişken: Tax_PIT
   • Açıklayıcı Göstergeler: Salary_Inx (Maaş Endeksi), Employee_Vol (Çalışan Hacmi), Deposit_Vol (Mevduat Hacmi), Deposit_Int_Rt (Mevduat Faiz Oranı).
   • Aktif Uzman Kuralı: Toplam gelir vergisi içinde ücretlerin payı %54, Menkul Sermaye İratlarının (MSİ) payı %18 olarak kabul edilmiştir. Sistem, ücret indikatörlerinin (Salary_Inx, Employee_Vol) trendi ile MSİ indikatörlerinin (Deposit_Vol, Deposit_Int_Rt) trendini bu oranlarda ağırlıklandırarak makroekonomik bir beklenti oluşturur ve sinir ağının ham tahminini bu beklenti ekseninde dengeler. İndikatör kısıtından dolayı ölçülemeyen %26'lık kör noktayı bütçe paylarına göre ağırlıklandırarak ham tahmini rasyonelleştirir.

2. Kurumlar Vergisi (Tax_CIT) - Çeyreklik Veri (n / n-4)
   • Bağımlı Değişken: Tax_CIT
   • Açıklayıcı Göstergeler: CPI_Inx (TÜFE), Man_Prod_Inx (İmalat Sanayi Üretim Endeksi), Trade_Vol_Inx (Ticaret Hacim Endeksi), Reel_Sector_Tax_Rt (Reel Sektör Vergi Oranı), Banking_Int_Rate_Margin (Banka Faiz Marjı), Banking_Sector_Tax_Rt (Bankacılık Vergi Oranı).
   • Aktif Uzman Kuralı: Toplam kurumlar vergisi içinde Reel Sektörün payı %75, Bankacılık/Finans sektörünün payı %25 olarak kabul edilmiştir. Sistem; Reel Sektör indikatör grubu (TUFE_Inx, Man_Prod_Inx, Trade_Vol_Inx, Reel_Sector_Tax_Rt) ile Bankacılık sektörü indikatör grubunun (Banking_Int_Rate_Margin, Banking_Sector_Tax_Rt) değişim trendlerini %75-%25 ağırlık rasyolarıyla birleştirerek yasal oran değişikliklerine ve kârlılık şoklarına duyarlı bir düzeltme mekanizması uygular.

3. İthalatta Alınan KDV (Tax_Import_VAT) - Aylık Veri (n / n-12)
   • Bağımlı Değişken: Tax_Import_VAT
   • Açıklayıcı Göstergeler: Import_ExcGOLD_USD_Vol (Altın Hariç İthalat Dolar Hacmi), Ave_Dolar_Exc_Rt (Ortalama Dolar Kuru), Import_VAT_Effective_Tax_Rt (Efektif İthalat KDV Oranı).
   • Aktif Uzman Kuralı:

4. Dahili KDV (Tax_Domestic_VAT) - Aylık Veri (n / n-12)
   • Bağımlı Değişken: Tax_Domestic_VAT
   • Açıklayıcı Göstergeler: CPI_Inx (TÜFE), Trade_Vol_Inx (Ticaret Hacmi), CreditCart_Usage_Vol (Kredi Kartı Kullanım Hacmi), Tax_Total_SCT (Toplam ÖTV Gelirleri).
   • Aktif Uzman Kuralı:

5. Petrol ve Doğalgaz ÖTV'si (Tax_Petrelum_SCT) - Aylık Veri (n / n-12)
   • Bağımlı Değişken: Tax_Petrelum_SCT
   • Açıklayıcı Göstergeler: Cons_Gasoline (Benzin Tüketimi), Cons_Diesel (Motorin Tüketimi), Fix_Tax_Gasoline (Benzin Maktu ÖTV), Fix_Tax_Diesel (Motorin Maktu ÖTV).
   • Aktif Uzman Kuralı:

6. Tütün ÖTV'si (Tax_Tobacco_SCT) - Aylık Veri (n / n-12)
   • Bağımlı Değişken: Tax_Tobacco_SCT
   • Açıklayıcı Göstergeler: Cons_Tobacco (Tütün Tüketimi), Advolorem_Tax_Tobacco (Nispi ÖTV Oranı).
   • Aktif Uzman Kuralı:

7. Motorlu Taşıtlar ÖTV'si (Tax_Motor_Vehicle_SCT) - Aylık Veri (n / n-12)
   • Bağımlı Değişken: Tax_Motor_Vehicle_SCT
   • Açıklayıcı Göstergeler: Motor_Vehicle_Sales_Vol, Motor_Vehicle_Average_Tax_Base, Motor_Vehicle_Effective_Tax_Rt, Electrical_Vehicle_Sales_Vol.
   • Aktif Uzman Kuralı:

# 5. KURULUM VE ÇALIŞTIRMA TALİMATLARI

# ⚡ 5.1 NVIDIA GPU Hızlandırması Kurulum Kılavuzu (Opsiyonel)

Bu projeyi işlemci (CPU) yerine NVIDIA ekran kartınızın (GPU) tensör çekirdeklerini kullanarak çok daha hızlı eğitmek isterseniz, sisteminize **CUDA Toolkit** ve **cuDNN** kütüphanelerini kurmanız gerekmektedir.

> ⚠️ **Önemli Not:** Yeni nesil NVIDIA mimarileri (Örn: Blackwell / RTX 50 serisi vb.) için minimum **CUDA 12.x** ve üzeri sürümlerin kurulması zorunludur.

# 5.1.1. NVIDIA Ekran Kartı Sürücüsü Güncelleme

CUDA kurulumuna geçmeden önce ekran kartı sürücünüzün en güncel sürümde olduğundan emin olun.

- **Resmi Sürücü İndirme Linki:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

# 5.1.2. NVIDIA CUDA Toolkit Kurulumu

İşletim sisteminize (Windows/Linux) uygun olan güncel CUDA Toolkit sürümünü indirin ve "Network" veya "Local" yükleyici tipini seçerek standart kurulumu tamamlayın.

- **Resmi CUDA İndirme Linki:** [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- _Önerilen Sürüm:_ TensorFlow ve PyTorch kararlılığı için **CUDA 12.1** veya **CUDA 12.4** tercih edebilirisiniz.

# 5.1.3. NVIDIA cuDNN (CUDA Deep Neural Network) Kurulumu

Derin sinir ağlarındaki matris çarpımlarının GPU tarafından donanımsal olarak hızlandırılmasını sağlayan alt kütüphanedir. Kurduğunuz CUDA sürümüyle uyumlu olan cuDNN paketini indirin.

- **Resmi cuDNN İndirme Linki:** [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn)

> 💡 **Kurulum İpucu (Windows):** cuDNN indikten sonra zip dosyasının içindeki `bin`, `include` ve `lib` klasörlerinin içeriğini, CUDA'nın bilgisayarınızda kurulu olduğu dizine (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`) kopyalayıp yapıştırın.

# 5.1.4. Ortam Değişkenlerinin (Environment Variables) Kontrolü

Kurulumların ardından terminalin (CMD) CUDA'yı tanıyabilmesi için şu yolların sistem ortam değişkenlerindeki `Path` kısmına eklendiğinden emin olun:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp

# 5.2 Sanal ortamı ayağa kaldırmak, bağımlılıkları yüklemek ve projeyi çalıştırmak için kök dizinde sırasıyla şu terminal komutlarını uygulayın:

# 5.2.1. Sanal ortam kurulumu ve aktivasyon

python -m venv venv
source venv/bin/activate # Windows kullanıyorsanız: .\venv\Scripts\activate

# 5.2.2 Gerekli kütüphane paketlerinin yüklenmesi

python -m pip install --upgrade pip
pip install -r requirements.txt

# 5.2.3. Projenin uçtan uca çalıştırılması ve MAE raporlaması

python main.py
