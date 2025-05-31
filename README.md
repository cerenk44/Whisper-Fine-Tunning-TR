# Whisper-Fine-Tunning-TR
# Türkçe Konuşma Tanıma için Whisper Modelinin İyileştirilmesi (Fine-tuning)

Bu proje, OpenAI'nin Whisper modelinin "small" versiyonunu kullanarak Türkçe Otomatik Konuşma Tanıma (ASR) sistemi geliştirmeyi amaçlamaktadır. Model, Mozilla Common Voice 11.0 Türkçe veri kümesi üzerinde Hugging Face Transformers kütüphanesi kullanılarak iyileştirilmiştir (fine-tuned).

##  İçindekiler

* [Proje Hakkında](#proje-hakkında)
* [Kullanılan Veri Seti](#kullanılan-veri-seti)
* [Model Detayları](#model-detayları)
* [Kurulum](#kurulum)
* [Kullanım](#kullanım)
    * [Eğitim](#eğitim)
    * [Test ve Değerlendirme](#test-ve-değerlendirme)
    * [Gradio Arayüzü ile Çıkarım](#gradio-arayüzü-ile-çıkarım)
* [Kod Yapısı](#kod-yapısı)
* [Önemli Hiperparametreler](#önemli-hiperparametreler)
* [Sonuçlar (Örnek)](#sonuçlar-örnek)
* [Teşekkür](#teşekkür)

## Proje Hakkında

Bu projenin temel amacı, `openai/whisper-small` modelini Türkçe konuşma verileriyle eğiterek, Türkçe için yüksek performanslı bir konuşma tanıma modeli elde etmektir. Proje, Google Colab üzerinde geliştirilmiş ve Hugging Face ekosisteminden (Transformers, Datasets, Evaluate) yoğun bir şekilde faydalanılmıştır.

**Temel Özellikler:**
* Türkçe konuşmayı metne çevirme.
* Whisper-small modelinin fine-tune edilmiş versiyonu.
* Common Voice 11.0 Türkçe veri seti ile eğitim.
* Eğitim ve test süreçleri için Colab not defterleri.
* Modeli interaktif olarak test etmek için Gradio arayüzü.

## Kullanılan Veri Seti

Modelin eğitimi ve değerlendirilmesi için [Mozilla Common Voice 11.0](https://commonvoice.mozilla.org/) veri setinin Türkçe (`tr`) bölümü kullanılmıştır.
* **Eğitim için:** `train` ve `validation` birleştirilmiş split'inden bir alt küme (örneğin ilk 2000 örnek).
* **Test için:** `test` split'i (örneğin ilk 800 örnek veya tamamı).

Veri ön işleme adımları şunları içerir:
* Gereksiz sütunların kaldırılması (accent, age, client_id vb.).
* Ses örnekleme oranının 16kHz'e dönüştürülmesi.

## Model Detayları

* **Temel Model:** `openai/whisper-small`
* **İyileştirilmiş Model (Hugging Face Hub):** `aysunn/whisper-small-tr`
* **Dil:** Türkçe
* **Görev:** Otomatik Konuşma Tanıma (Transkripsiyon)

## Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/](https://github.com/)cerenk44/Whisper-Fine-Tunning-TR.git
    cd Whisper-Fine-Tunning-TR
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio torch
    ```

3.  **Hugging Face Hub Girişi (Gerekirse):**
    Modeli Hub'a yüklemek veya özel modelleri indirmek için Hugging Face hesabınıza giriş yapmanız gerekebilir.
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```
    Bu komutu Python veya IPython ortamında çalıştırın ve token'ınızı girin.

## Kullanım

Proje, Google Colab not defterleri (`.ipynb`) olarak sağlanmıştır. Bu not defterlerini Google Colab'da açarak veya yerel Jupyter ortamınızda çalıştırabilirsiniz.

### Eğitim

1.  **Eğitim Not Defteri:** `OtomatikAltyazıEkleme.ipynb`
2.  Not defterini açın ve hücreleri sırayla çalıştırın.
3.  **Önemli Ayarlar:**
    * `small_dataset = common_voice["train"].select(range(2000))`: Eğitim için kullanılacak örnek sayısını buradan ayarlayabilirsiniz.
    * `Seq2SeqTrainingArguments`: Bu bölümdeki hiperparametreleri (batch_size, learning_rate, max_steps vb.) kendi ihtiyaçlarınıza veya kaynaklarınıza göre düzenleyebilirsiniz.
    * `output_dir`: Modelin ve çıktıların kaydedileceği yer.
    * `push_to_hub=True` ve `hub_model_id`: Eğer modeli kendi Hugging Face Hub hesabınıza yüklemek istiyorsanız bu kısımları uygun şekilde doldurun.
4.  Eğitim sonunda model, belirttiğiniz `output_dir`'e ve (eğer ayarlandıysa) Hugging Face Hub'a yüklenecektir.

### Test ve Değerlendirme

1.  **Test Not Defteri:** `test.ipynb` (veya kendi verdiğiniz isim)
2.  Not defterini açın ve hücreleri sırayla çalıştırın.
3.  **Önemli Ayarlar:**
    * `dataset_name = "mozilla-foundation/common_voice_11_0"` ve `language_code = "tr"`: Kullanılacak veri setini belirtir.
    * `model_name_or_path = "aysunn/whisper-small-tr"`: Değerlendirilecek modelin Hugging Face Hub'daki adını veya yerel yolunu buraya girin.
    * `ISTENEN_ALT_KUME_BOYUTU = 800`: Hızlı test için kullanılacak örnek sayısı. Tüm test seti için bu sayıyı veri seti boyutuna göre ayarlayın veya daha büyük tutun.
4.  Not defteri, modelin Kelime Hata Oranı (WER) skorunu hesaplayacak ve örnek tahminleri gösterecektir.

### Gradio Arayüzü ile Çıkarım

Eğitim not defterinin (`egitim_dosyasi.ipynb`) sonunda veya ayrı bir script'te bulunan Gradio arayüzü ile eğittiğiniz modeli interaktif olarak test edebilirsiniz.

1.  Pipeline'da model adını kendi modelinizle güncelleyin:
    ```python
    pipe = pipeline(model="aysunn/whisper-small-tr")
    ```
2.  Gradio arayüzünü başlatmak için ilgili hücreleri çalıştırın:
    ```python
    iface.launch(debug=True) 
    ```
3.  Açılan link üzerinden mikrofonla konuşarak veya ses dosyası yükleyerek modeli deneyebilirsiniz.

## Kod Yapısı

* `OtomatikAltyazıEkleme.ipynb`: Whisper modelini Türkçe Common Voice veri seti ile fine-tune etmek için kullanılan Colab not defteri. Veri yükleme, ön işleme, model yapılandırma, eğitim ve modeli Hugging Face Hub'a yükleme adımlarını içerir. Ayrıca bir Gradio demosu da barındırır.
* `test.ipynb`: Fine-tune edilmiş modelin performansını Common Voice test seti üzerinde değerlendirmek için kullanılan Colab not defteri. WER metriğini hesaplar ve örnek tahminler sunar.

##  Önemli Hiperparametreler

Eğitim sürecinde `Seq2SeqTrainingArguments` içerisinde ayarlanan bazı önemli hiperparametreler ve nedenleri:

* `per_device_train_batch_size = 4`, `gradient_accumulation_steps = 4`: Efektif batch boyutu 16. Colab GPU belleği kısıtlamaları nedeniyle gradyan biriktirme kullanıldı.
* `learning_rate = 1e-5`: Fine-tuning için yaygın ve genellikle iyi sonuç veren bir öğrenme oranı.
* `warmup_steps = 100`: Eğitimin başında öğrenme oranını yavaşça artırarak stabilizasyon sağlar.
* `max_steps = 2000`: Toplam eğitim adımı sayısı. Kullanılan veri kümesi boyutuna ve istenen eğitim süresine göre ayarlandı.
* `fp16 = True`: Karma hassasiyetli eğitim ile hız ve bellek optimizasyonu.
* `eval_strategy = "steps"`, `eval_steps = 500`: Belirli adımlarda model performansını izlemek için.
* `load_best_model_at_end = True`, `metric_for_best_model = "wer"`: En iyi modeli WER'e göre seçmek için.

## Sonuçlar (Örnek)

Fine-tune edilmiş `whisper-small-tr` modeli, Common Voice 11.0 Türkçe test seti üzerinde **%XX.XX WER** (Kelime Hata Oranı) elde etmiştir.

Örnek Tahminler:
*(Test script'inden birkaç iyi ve/veya zorlu örnek transkripsiyonu buraya ekleyebilirsiniz.)*
|  Referans Cümle                                      | Model Tahmini                                        |
|  Bana bir şey olmaz.                                 | Fana bir şey olmaz..                                 |
|  Broz, büyükbabasının başarılarını gururla anıyor.   | Buros, büyük babasının başarılarını gururla a        |

## Teşekkür

* Bu çalışmada kullanılan [Mozilla Common Voice](https://commonvoice.mozilla.org/) veri setini sağlayanlara.
* [OpenAI](https://openai.com/)'ye Whisper modelini geliştirdikleri için.
* [Hugging Face](https://huggingface.co/) ekibine, Transformers, Datasets ve diğer harika araçları için.
