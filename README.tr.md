# WBC Analyzer: Yapay Zeka Destekli Patoloji Asistanı

<p align="center">
  <img src="docs/banner.png" alt="WBC Analyzer Banner" width="100%">
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> &nbsp;|&nbsp; 🇹🇷 Türkçe
</p>

Periferik kan yayma görüntülerinden otomatik beyaz kan hücresi (WBC) sınıflandırması için uçtan uca derin öğrenme sistemi — ajansal LLM açıklanabilirliği ile birlikte üretime hazır Flask REST API olarak dağıtılmıştır.

---

## Sonuçlar

| Set                      | n     | Doğruluk   | Ağırlıklı F1 |
| ------------------------ | ----- | ---------- | ------------ |
| **TestA** (dağılım içi)  | 4.339 | **98,53%** | **0,9854**   |
| **TestB** (alan kayması) | 2.119 | **89,05%** | **0,9111**   |
| **Birleşik**             | 6.458 | **95,42%** | **0,9554**   |

TestB, farklı bir mikroskoptan elde edilmiş yalnızca iki sınıf (Lenfosit, Nötrofil) içermektedir — standart doğruluk değil, cihazlar arası genelleme yeteneğini ölçer. Çıkarım zamanı adaptasyonu olmadan temel başarı: **%56,96**. Tam pipeline sonrası kazanım: **+32,09 puan**.

**Sınıf bazlı performans (TestA):**

| Sınıf     | Kesinlik | Duyarlılık | F1     | Örnek Sayısı |
| --------- | -------- | ---------- | ------ | ------------ |
| Bazofil   | 1,0000   | 1,0000     | 1,0000 | 89           |
| Eozinofil | 0,9265   | 0,9783     | 0,9517 | 322          |
| Lenfosit  | 0,9865   | 0,9884     | 0,9874 | 1.034        |
| Monosit   | 0,9372   | 0,9573     | 0,9471 | 234          |
| Nötrofil  | 0,9962   | 0,9868     | 0,9915 | 2.660        |

---

## Mimari

**Omurga:** DenseNet121 (7,70 M parametre, 1. Aşamada dondurulmuş)

**Özgün bileşenler:**

- `WBCAttentionBlock` — lökosit morfolojisine uyarlanmış CBAM tarzı kanal + mekansal dikkat (132.259 parametre)
- `MedSwish` — α ve β parametrelerine sahip öğrenilebilir aktivasyon; ince morfolojik detaylarda Dying ReLU'yu bastırır (4 parametre)
- `WBCFocalLoss` — sınıf dengesizliğini yönetmek için sınıfa özel ağırlıklı focal loss (Bazofil: nadir; Nötrofil: baskın)
- Ana 5 sınıflı kafa ile ortaklaşa eğitilen yardımcı ikili kafa (`Nötrofil - Lenfosit`)

**Toplam eğitilebilir parametre: 7,83 M** (VGG16'nın 138 M parametresinin ~%6'sı)

**Ön İşleme — Medical Enhanced Filter (MEF, 5 adım):**

1. Yüzdelik tabanlı renk normalizasyonu (kanal başına 2.–98. yüzdelik)
2. LAB uzayında çift ölçekli CLAHE (çekirdek için 4×4 + sitoplazma için 8×8 tile, Canny kenar ağırlıklarıyla birleştirilir)
3. Kenar korumalı bilateral filtre (d=9, σ_c=65, σ_s=65)
4. Morfolojik çekirdek vurgulama (iç k3×3 + dış k7×7 gradyan karışımı)
5. Seçici LoG keskinleştirme (yalnızca kenarlarda; düzgün bölgeler korunur)

**Çıkarım zamanı alan adaptasyonu (yeniden eğitim gerektirmez):**

| Adım                                         | TestB Δ                 |
| -------------------------------------------- | ----------------------- |
| Adaptasyon yok (temel)                       | %56,96                  |
| + İkili yönlendirme (main_out)               | +16,94 puan → %73,90    |
| + Reinhard renk normalizasyonu               | +12,56 puan → %86,46    |
| + Hafif TTA (çevirme + döndürme + parlaklık) | +2,59 puan → **%89,05** |

**Omurga karşılaştırması (doğrulama seti, aynı eğitim protokolü):**

| Model                                     | Parametre (M) | Val Doğruluk (%) | Macro F1   | Çıkarım (ms) |
| ----------------------------------------- | ------------- | ---------------- | ---------- | ------------ |
| VGG16                                     | 15,11         | 98,56            | 0,9724     | 18,1         |
| ResNet50V2                                | 24,75         | 98,17            | 0,9704     | 103,9        |
| MobileNetV2                               | 3,05          | 97,90            | 0,9577     | 96,0         |
| EfficientNetB0                            | 4,84          | 97,05            | 0,9418     | 185,4        |
| DenseNet121 (yalın)                       | 7,70          | **98,89**        | **0,9803** | 232,2        |
| **DenseNet121 + WBCAttention + MedSwish** | **7,83**      | 98,53            | 0,9853     | **14,2**     |

---

## Ajansal XAI

Sistem iki katmanlı bir kısayol öğrenme koruma mekanizması çalıştırır:

**Eğitim katmanı — XAIFocusMonitor callback'i:**

- Her N epoch'ta doğrulama seti üzerinde Grad-CAM ön plan odak oranını (ρ) hesaplar
- ρ değeri `--xai-patience` kadar ardışık kontrolde eşiğin (varsayılan 0,55) altına düşerse eğitimi erken sonlandırır
- Arka plan kısayol öğrenmesini insan müdahalesine gerek kalmadan özerk biçimde tespit eder

**Çıkarım katmanı — LLM ajanı:**

- Birincil: GitHub Models üzerinden `openai/gpt-4o`
- Yedek: Google GenAI SDK üzerinden `gemini-2.5-flash`
- Her iki API kullanılamadığında kural tabanlı yedek
- Grad-CAM ısı haritası kaplaması + hücre tipine özel morfolojik bağlam istemi → özerk klinik açıklama raporu

---

## Depo Yapısı

```
wbc-final/
├── app.py                        # Flask REST API + LLM ajanı
├── train_main_model.py           # Ana model eğitimi (Aşama 1 + Aşama 2 + XAI izleme)
├── train_baseline_comparison.py  # 5 omurga karşılaştırmalı eğitim
├── eval_final.py                 # TTA + ikili yönlendirme + Reinhard ile değerlendirme
├── eval_baseline.py              # Temel omurga sonuçları için değerlendirme
├── src/
│   ├── custom_layers.py          # WBCAttentionBlock, MedSwish
│   ├── custom_losses.py          # WBCFocalLoss
│   └── preprocessing.py         # MEF + Reinhard normalizasyonu (v1–v4 varyantları)
├── data/
│   ├── models/                   # .keras model dosyasını buraya yerleştirin
│   └── raabin-wbc-data/          # Veri seti (Train / TestA / TestB)
├── outputs/
│   ├── final_model_results/      # Sınıflandırma raporları, karmaşıklık matrisleri
│   └── baseline_results/         # Omurga karşılaştırma sonuçları
└── templates/index.html          # Web arayüzü
```

---

## Hızlı Başlangıç

**Gereksinimler:** Python 3.9+, TensorFlow 2.18, CUDA destekli GPU önerilir.

```bash
git clone https://github.com/frissonitte/wbc-analyzer-final.git
cd wbc-analyzer-final
pip install -r requirements.txt
```

[**Modeli indirin**](https://drive.google.com/file/d/1pV9vjLYF8KCilsxtkEmOaKwd25Dw57gZ/view?usp=sharing) ve şu konuma yerleştirin:

```
data/models/wbc_final_model_densenet.keras
```

API anahtarlarınızla `.env` dosyası oluşturun:

```env
GITHUB_TOKEN=github_models_tokeniniz
GEMINI_API_KEY=gemini_api_anahtariniz
```

Web uygulamasını çalıştırın:

```bash
python app.py
```

`http://localhost:5000` adresini açın, bir WBC görüntüsü sürükleyip bırakın ve sınıflandırma + Grad-CAM + LLM raporu alın.

---

## En İyi Sonuçları Yeniden Üretin

Eğitilmiş modeli tam çıkarım zamanı adaptasyon pipeline'ıyla (Reinhard + ikili yönlendirme + hafif TTA) değerlendirin:

```bash
python eval_final.py \
    --model-path data/models/wbc_final_model_densenet.keras \
    --data-root data/raabin-wbc-data \
    --output-dir outputs/final_model_results \
    --testb-binary-mode main \
    --tta light \
    --color-normalization reinhard \
    --preprocessing v1
```

Çıktılar `--output-dir` konumuna kaydedilir: TestA / TestB / birleşik için `classification_report.txt`, `confusion_matrix.png`, `predictions.csv`.

---

## Sıfırdan Eğitim

**Ana model** (DenseNet121 + WBCAttention + MedSwish + XAI izleme):

```bash
python train_main_model.py \
    --data-root data/raabin-wbc-data \
    --phase1-epochs 15 \
    --phase2-epochs 15 \
    --main-loss cce \
    --label-smoothing 0.1 \
    --crop-prob 0.2 \
    --bg-randomization-prob 0.15 \
    --stain-jitter-prob 0.3 \
    --aux-loss-weight 1.0 \
    --xai-focus-threshold 0.55 \
    --xai-every-n-epochs 2 \
    --model-path data/models/wbc_final_model_densenet.keras
```

**Omurga karşılaştırması** (5 mimariyi özdeş koşullar altında eğitir):

```bash
python train_baseline_comparison.py \
    --data-root data/raabin-wbc-data \
    --results-dir outputs/baseline_results
```

Azaltılmış epoch sayısıyla deneme çalıştırması için `--fast`, belirli modelleri eğitmek için `--models VGG16 DenseNet121_vanilla` ekleyin.

---

## Veri Seti

[Raabin-WBC](https://raabin.ir/) — Tahran Üniversitesi Tıp Bilimleri tarafından yayımlanan büyük açık erişimli veri seti.  
5 sınıf: Bazofil, Eozinofil, Lenfosit, Monosit, Nötrofil.  
Giemsa boyamalı periferik kan yayma görüntüleri hem akıllı telefon kameralarıyla (Samsung S5) hem de profesyonel mikroskop kameralarıyla elde edilmiştir — iki cihazlı kurulum, bu projede ele alınan alan genellemesi zorluğunu yaratmaktadır.

- Eğitim: ~12.000 görüntü
- TestA: 4.339 görüntü (5 sınıf, aynı cihaz dağılımı)
- TestB: 2.119 görüntü (2 sınıf: Lenfosit + Nötrofil, farklı cihaz)

---

## Ön İşleme Ablasyon Analizi

Aynı eğitilmiş model, dört farklı ön işleme varyantıyla test edilmiştir:

| Varyant                                                       | TestA      | TestB      | Birleşik   |
| ------------------------------------------------------------- | ---------- | ---------- | ---------- |
| v1 — MEF orijinal (clip + CLAHE + bilateral + keskinleştirme) | **%98,41** | %85,65     | %94,22     |
| v2 — Adaptif CLAHE tileGrid (8×8)                             | %97,99     | **%87,92** | **%94,69** |
| v3 — v2 + top-hat / bottom-hat morfoloji                      | %95,18     | %77,58     | %89,41     |
| v4 — v3 + Macenko boyama normalizasyonu (kalibre edilmemiş)   | %57,78     | %42,28     | %52,69     |

v4'teki çöküş, Macenko'nun veri setine özgü referans matrisi olmadan uygulanmasından kaynaklanmaktadır. En iyi TestA/Birleşik dengesini sağladığından nihai değerlendirmede v1 kullanılmaktadır.

---

## API Referansı

`POST /predict`

| Alan   | Tür                 | Açıklama                                  |
| ------ | ------------------- | ----------------------------------------- |
| `file` | multipart/form-data | WBC görüntüsü (JPG, PNG, BMP, TIFF, WebP) |

**Yanıt (200):**

```json
{
  "class": "Neutrophil",
  "confidence": 0.977,
  "all_probabilities": {...},
  "gradcam_image": "<base64>",
  "llm_report": "Grad-CAM aktivasyonu çekirdek lob yapısına odaklandı..."
}
```

**Hata kodları:** `400` hatalı görüntü · `415` desteklenmeyen format · `500` model hatası

---

## Yazar

Emirhan Yıldırım  
[emirhan.yildirim2@ogr.sakarya.edu.tr](mailto:emirhan.yildirim2@ogr.sakarya.edu.tr)  
Sakarya Üniversitesi — Bilişim Sistemleri Mühendisliği  
ISE 402 Bitirme Projesi · 2025–2026 Bahar Dönemi
