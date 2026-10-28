# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

**Öğrenci Adı Soyadı:** Furkan Dinç  
**Öğrenci Numarası:** 23120205051  

---

## Proje Hakkında

Bu proje, MNIST el yazısı rakam veri setini kullanarak sıfırdan bir Çok Katmanlı Algılayıcı (MLP - Multi-Layer Perceptron) modeli eğitmek ve modelin başarımını hem MNIST test verisi hem de kendi el yazımla oluşturduğum öğrenci numaram (**23120205051**) üzerindeki rakamlar ile değerlendirmek amacıyla hazırlanmıştır.

---

## 1. Model Mimarisi ve Hiperparametreler

Ödev kuralları gereği **Convolutional Neural Network (CNN) kullanılnamış**, tamamen Tam Bağlı (Fully Connected / Dense) katmanlardan oluşan bir MLP mimarisi tercih edilmiştir.

* **Giriş Katmanı:** 784 Düğüm ($28 \times 28$ piksel düzleştirilmiş vektör)
* **1. Gizli Katman (Hidden Layer 1):** 256 Nöron + ReLU Aktivasyonu
* **2. Gizli Katman (Hidden Layer 2):** 128 Nöron + ReLU Aktivasyonu
* **Çıkış Katmanı:** 10 Nöron (0–9 arası sınıflar)
* **Kayıp Fonksiyonu (Loss Function):** CrossEntropyLoss
* **Optimizasyon Algoritması:** Adam (`learning_rate = 0.001`)
* **Batch Size:** 64
* **Epoch Sayısı:** 10
* **Veri Kümesi Ayrımı:** 50.000 Eğitim (Train), 10.000 Doğrulama (Validation), 10.000 Test

---

## 2. Eğitim Sonuçları ve Performans Metrikleri

Model PyTorch kütüphanesi kullanılarak eğitilmiştir.

* **Eğitim Süresi:** 49.70 saniye[cite: 13]
* **Son Eğitim Kaybı (Train Loss):** 0.0202[cite: 13]
* **Son Eğitim Başarısı (Train Acc):** %99.29[cite: 13]
* **Son Doğrulama Kaybı (Val Loss):** 0.1116[cite: 13]
* **Son Doğrulama Başarısı (Val Acc):** %97.59[cite: 13]
* **Test Başarısı (Test Accuracy):** **%97.67**[cite: 13]

Eğitim sürecine ait Training/Validation Loss ve Accuracy grafik değişimi `kendi_el_yaziniz_sonuclari.png` dosyasında görselleştirilmiştir[cite: 13].

---

## 3. Kendi El Yazısı Görüntüleri ve Dağılım Kayması (Distribution Shift) Analizi

Öğrenci numaram olan **23120205051** sayısının her bir rakamı (`2`, `3`, `1`, `2`, `0`, `2`, `0`, `5`, `0`, `5`, `1`) kareli kağıda yazılıp fotoğraflanmış, görüntüler $28 \times 28$ piksele boyutlandırılarak renkleri MNIST formatına uygun şekilde tersine çevrilmiştir (Inversion).

### Değerlendirme ve Genelleyebilme Yorumu:
1. **Veri Dağılımı Farkı (Distribution Shift):** MNIST veri seti pürüzsüz, gürültüsüz siyah arka plana ve ortalanmış beyaz rakamlara sahiptir. Gerçek hayatta çekilen fotoğraflarda ise kareli kağıt çizgileri ve ortam ışığından kaynaklanan gölgelenmeler yer almaktadır.
2. **Model Davranışı:** Renk ters çevirme (inversion) işlemi yapıldığında, kağıttaki koyu renkli kare çizgileri piksel seviyesinde yüksek değerler alarak arka planda izler oluşturmuştur.
3. **Sonuç:** MLP modeli mekânsal/konumsal ilişkileri CNN gibi evrişim filtreleriyle öğrenmeyip sadece düzleştirilmiş piksel değerlerine baktığı için bu arka plan gürültülerinden doğrudan etkilenmiştir. Bu durum, eğitim verisi ile gerçek dünya verisi arasındaki dağılım farkının (generalization gap) model başarımına olan etkisini açıkça göstermektedir.