# Laporan Proyek Machine Learning - Cindy Maharani

## Domain Proyek

Stroke adalah salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Deteksi dini terhadap risiko stroke sangat penting untuk pencegahan dan penanganan cepat. Gejala-gejala seperti nyeri dada, sesak napas, detak jantung tidak teratur, dan tekanan darah tinggi merupakan indikator penting yang dapat digunakan untuk memprediksi risiko stroke seseorang. Dengan menggunakan teknik machine learning, kita dapat membangun model prediksi berbasis gejala untuk mengidentifikasi individu yang berisiko lebih tinggi mengalami stroke.

Referensi: [Penanganan Kegawatdaruratan Di Rumah: Serangan Stroke dan Pencegahan Terjadinya Stroke](https://www.journalmpci.com/index.php/jppmi/article/view/40) 

## Business Understanding

### Problem Statements
- Bagaimana cara mengklasifikasikan individu berdasarkan gejala-gejala yang mereka alami menjadi 'At Risk' atau 'Not At Risk' terkena stroke?
- Seberapa akurat prediksi risiko stroke menggunakan gejala klinis sederhana tanpa memerlukan pemeriksaan medis lanjutan?

### Goals
- Membuat model machine learning untuk memprediksi risiko stroke (At Risk: 0/1) berdasarkan gejala-gejala yang dilaporkan.
- Menilai performa model berdasarkan metrik klasifikasi seperti accuracy, precision, recall, dan F1-score.

### Solution statements
- Menggunakan dua model machine learning: Logistic Regression (sebagai baseline) dan Random Forest Classifier (sebagai model yang lebih kompleks).
- Memilih model terbaik berdasarkan nilai F1-score, mengingat pentingnya keseimbangan antara precision dan recall dalam kasus medis.

## Data Understanding
Dataset ini dirancang untuk mendukung penelitian prediksi risiko stroke menggunakan metode machine learning dan deep learning. Dataset ini berisi 7000 baris dengan 18 kolom. 

Dataset disusun berdasarkan literatur medis, konsultasi dengan para ahli, serta pemodelan statistik, dan mengacu pada sumber terpercaya seperti American Stroke Association (ASA), Mayo Clinic, WHO, hingga buku teks medis seperti Harrison’s Principles of Internal Medicine.

Dataset yang digunakan dalam penelitian ini dapat diakses melalui tautan berikut: [Stroke Risk Prediction Dataset Based on Symptoms](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset).

### Variabel-variabel pada dataset:
- Chest Pain: 0/1, nyeri dada
- Shortness of Breath: 0/1, sesak napas
- Irregular Heartbeat: 0/1, detak jantung tidak teratur
- Fatigue & Weakness: 0/1, kelelahan dan kelemahan
- Dizziness: 0/1, pusing
- Swelling (Edema): 0/1, pembengkakan
- Pain in Neck/Jaw/Shoulder/Back: 0/1, nyeri di leher/rahang/bahu/punggung
- Excessive Sweating: 0/1, keringat berlebih
- Persistent Cough: 0/1, batuk kronis
- Nausea/Vomiting: 0/1, mual/muntah
- High Blood Pressure: 0/1, tekanan darah tinggi
- Chest Discomfort (Activity): 0/1, ketidaknyamanan di dada saat aktivitas
- Cold Hands/Feet: 0/1, tangan/kaki terasa dingin
- Snoring/Sleep Apnea: 0/1, mendengkur/apnea tidur
- Anxiety/Feeling of Doom: 0/1, kecemasan atau firasat buruk
- Stroke Risk (%): 0-100%, persentase estimasi risiko stroke
- At Risk: 0/1, target label (1 = berisiko, 0 = tidak berisiko)
- Age: Usia individu
Catatan:
- Fitur target untuk prediksi adalah At Risk.
- Fitur Stroke Risk (%) tidak digunakan dalam prediksi untuk menghindari data leakage.

### EDA
#### Distribusi Kolom Target (At Risk)
![Distribusi At Risk](https://drive.google.com/file/d/1aptmRf1-ozb63YT-S6qRChfVA6F8YmiK/view?usp=sharing)
Kolom "At Risk (Binary)" merupakan label klasifikasi yang menentukan apakah seseorang berisiko terkena stroke atau tidak berdasarkan berbagai faktor klinis dan gejala. Meskipun distribusi kelas menunjukkan sedikit ketidakseimbangan, namun proporsinya masih dapat diterima untuk melatih model klasifikasi tanpa perlakuan khusus balancing di tahap awal.

#### Umur Vs Stroke Risk
![Stroke Risk Vs Age](https://drive.google.com/file/d/1DIj8MUc7qqaLVj5Sqq4A34kL9LrPFLA7/view?usp=drive_link)
Dapat dilihat dari visualisasi dengan scatter plot bahwa semakin tua usia seseorang maka persentase resiko terkena stroke juga semakin tinggi. Persentase terkena stroke diatas 50% dapat dikategorikan sebagai 'At Risk' atau Beresiko Stroke.

## Data Preparation
Langkah–langkah data preparation yang dilakukan.
1. Cek informasi ringkas tentang dataframe dengan df.info()
Untuk melihat tipe data dan cek apakah ada data yang kosong/ missing. Pada dataset ini tidak terdapat missing value.
2. Tangani kolom duplikat
Dataset ini memiliki 1021 baris yang duplikat. Baris yang duplikat tersebut dihapus untuk meingkatkan kualitas data.
3. Feature Selection
Kolom Stroke Risk (%) di-drop karena mengandung informasi target yang seharusnya diprediksi. Kolom ini dipertahankan jika ingin melakukan tugas regresi.
4. Feature Scaling
Karena semua fitur bersifat biner (0/1) dan numerik (Age), scaling tidak wajib, tetapi dilakukan standar scaling pada Age untuk mempercepat konvergensi model Logistic Regression.
5. Train-Test Split
Dataset dibagi menjadi training set sebesar 80% dan test set sebesar 20%.

## Modeling
Pada tahap ini, dilakukan pembuatan model machine learning untuk memprediksi risiko stroke berdasarkan gejala dan usia pasien. Beberapa algoritma digunakan untuk dibandingkan performanya, yaitu:
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Logistic Regression
- XGBoost Classifier

### Tahapan dan Parameter yang Digunakan
1. Split Data
Data dibagi menjadi data latih dan data uji menggunakan metode train_test_split dengan rasio 80:20.
2. Feature Scaling
Untuk algoritma yang sensitif terhadap skala data (seperti SVM dan Logistic Regression), dilakukan scaling menggunakan StandardScaler.
3. Model Training
Setiap algoritma dilatih menggunakan data latih:
    - Random Forest Classifier: menggunakan parameter default (n_estimators=100, criterion="gini", dll).
    - Support Vector Machine (SVC): menggunakan parameter default (kernel='rbf').
    - Gaussian Naive Bayes: menggunakan parameter default.
    - Logistic Regression: menggunakan parameter default (penalty='l2', solver='lbfgs').
    - XGBoost Classifier: menggunakan parameter default (n_estimators=100, learning_rate=0.1, max_depth=3, dll).
4. Evaluasi Model
Setiap model dievaluasi menggunakan data uji berdasarkan akurasi, precision, recall, f1-score, dan ROC AUC.

### Kelebihan dan Kekurangan Setiap Algoritma
| Algoritma | Kelebihan | Kekurangan|
|-------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Random Forest | Tidak mudah overfitting; Menangani data tidak terstruktur dengan baik | Model kompleks dan berat untuk interpretasi; Membutuhkan banyak memori |
| Support Vector Machine (SVM) | Bagus untuk dataset kecil dengan margin yang jelas; Efektif untuk data high-dimensional | Kurang efektif untuk dataset besar; Sulit memilih kernel dan parameter optimal |
| Gaussian Naive Bayes | Cepat dan efisien; Performa bagus di dataset kecil dan sederhana     | Asumsi independensi antar fitur sering tidak realistis; Kurang akurat untuk data kompleks |
| Logistic Regression | Sederhana, cepat, dan mudah diinterpretasikan; Cocok untuk klasifikasi biner | Tidak cocok untuk data dengan hubungan non-linear tanpa teknik tambahan |
| XGBoost Classifier| - Akurasi tinggi; Memiliki regularisasi untuk mencegah overfitting     | Waktu training lebih lama; Lebih kompleks dalam tuning hyperparameter |

### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi, model yang dipilih sebagai solusi adalah Logistic Regression.
Alasan pemilihan ini adalah:
- Logistic Regression memberikan akurasi dan f1-score yang lebih tinggi dibandingkan model lainnya.
- Cocok untuk klasifikasi biner.

## Evaluation
### Metrik yang digunakan:
1.  **Accuracy:**
    * **Formula:** $\frac{\text{Jumlah prediksi benar}}{\text{Total jumlah prediksi}} = \frac{TP + TN}{TP + FP + FN + TN}$
    * **Bagaimana Metrik Ini Bekerja:** *Accuracy* mengukur seberapa sering model Anda membuat prediksi yang benar, baik untuk pasien yang sebenarnya berisiko stroke (TP) maupun pasien yang sebenarnya tidak berisiko (TN), dibandingkan dengan total seluruh prediksi.
    * **Interpretasi:** Semakin tinggi nilai *accuracy*, semakin baik kinerja keseluruhan model dalam mengklasifikasikan risiko stroke dengan benar. Namun, perlu diingat bahwa *accuracy* bisa menyesatkan jika dataset Anda tidak seimbang (misalnya, jika jumlah pasien tidak berisiko jauh lebih banyak daripada yang berisiko).

2.  **Precision:**
    * **Formula:** $\frac{\text{Jumlah prediksi positif benar}}{\text{Total jumlah prediksi positif}} = \frac{TP}{TP + FP}$
    * **Bagaimana Metrik Ini Bekerja:** *Precision* mengukur dari semua pasien yang diprediksi berisiko stroke oleh model, berapa proporsi yang sebenarnya berisiko stroke. Ini berfokus pada seberapa tepat model dalam mengidentifikasi kelompok berisiko.
    * **Interpretasi:** *Precision* yang tinggi berarti bahwa ketika model memprediksi seorang pasien berisiko stroke, kemungkinan besar pasien tersebut memang berisiko. Metrik ini penting jika biaya *false positive* (memprediksi seseorang berisiko padahal tidak) tinggi, misalnya dalam hal kecemasan pasien atau biaya investigasi lanjutan yang tidak perlu.

3.  **Recall (Sensitivity atau True Positive Rate):**
    * **Formula:** $\frac{\text{Jumlah prediksi positif benar}}{\text{Total jumlah pasien positif sebenarnya}} = \frac{TP}{TP + FN}$
    * **Bagaimana Metrik Ini Bekerja:** *Recall* mengukur dari semua pasien yang sebenarnya berisiko stroke, berapa proporsi yang berhasil diidentifikasi dengan benar oleh model. Ini berfokus pada kemampuan model untuk tidak melewatkan kasus positif.
    * **Interpretasi:** *Recall* yang tinggi berarti model sangat baik dalam mendeteksi semua pasien yang berisiko stroke. Metrik ini sangat penting jika biaya *false negative* (gagal mengidentifikasi pasien yang berisiko) tinggi, karena hal ini bisa berakibat fatal bagi pasien.

4.  **F1-Score:**
    * **Formula:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
    * **Bagaimana Metrik Ini Bekerja:** *F1-Score* adalah rata-rata harmonik antara *precision* dan *recall*. Ini memberikan ukuran tunggal yang menyeimbangkan antara kemampuan model untuk menghindari *false positive* dan *false negative*. Rata-rata harmonik memberikan bobot yang lebih rendah pada nilai ekstrem, sehingga *F1-Score* akan rendah jika salah satu dari *precision* atau *recall* rendah.
    * **Interpretasi:** *F1-Score* yang tinggi menunjukkan bahwa model memiliki keseimbangan yang baik antara *precision* dan *recall*. Metrik ini berguna ketika Anda ingin menyeimbangkan kedua jenis kesalahan.

5.  **ROC AUC (Receiver Operating Characteristic Area Under the Curve):**
    * **Bagaimana Metrik Ini Bekerja:** ROC curve (kurva ROC) adalah grafik yang memplot *True Positive Rate* (TPR atau *recall*) terhadap *False Positive Rate* (FPR) pada berbagai *threshold* klasifikasi. FPR dihitung sebagai $\frac{FP}{FP + TN}$. AUC adalah area di bawah kurva ROC.
    * **Interpretasi:**
        * AUC = 0.5 berarti model tidak lebih baik dari tebakan acak.
        * AUC > 0.5 menunjukkan bahwa model memiliki kemampuan untuk membedakan antara kelas positif dan negatif.
        * AUC = 1.0 menunjukkan model yang sempurna dalam memisahkan kedua kelas.
        * ROC AUC mengukur kemampuan model untuk memberikan peringkat yang lebih tinggi pada instance positif dibandingkan instance negatif. Semakin tinggi nilai AUC, semakin baik model dalam membedakan antara pasien berisiko dan tidak berisiko, tanpa terpengaruh oleh pemilihan *threshold* klasifikasi tertentu.

### Hasil 

| Model                        | Accuracy | Precision | Recall   | F1-Score | ROC AUC |
| :--------------------------- | :------- | :-------- | :------- | :------- | :------ |
| Random Forest (RF)           | 0.945419 | 0.951762  | 0.964832 | 0.958252 | 0.937159 |
| Support Vector Machine (SVM) | 0.990287 | 0.991313  | 0.993748 | 0.992529 | 0.988814 |
| Naive Bayes (NB)             | 0.922296 | 0.930639  | 0.951211 | 0.940813 | 0.909993 |
| Logistic Regression          | 1.000000 | 1.000000  | 1.000000 | 1.000000 | 1.000000 |
| XGBoost                      | 0.996738 | 0.996435  | 0.998549 | 0.997491 | 0.995968 |

**Interpretasi Hasil:**

* **Logistic Regression:** Menunjukkan kinerja yang sempurna di semua metrik (1.000000). Ini mengindikasikan model mampu mengklasifikasikan risiko stroke dengan benar untuk semua sampel uji.
* **XGBoost:** Menunjukkan kinerja yang sangat tinggi di semua metrik, mendekati sempurna. Model ini sangat baik dalam memprediksi risiko stroke dengan keseimbangan yang baik antara *precision* dan *recall*.
* **Support Vector Machine (SVM):** Juga menunjukkan kinerja yang sangat baik dengan nilai metrik yang tinggi, menunjukkan kemampuan yang baik dalam memprediksi risiko stroke.
* **Random Forest (RF):** Menunjukkan kinerja yang baik, meskipun tidak sebaik tiga model teratas. Masih mampu memprediksi risiko stroke dengan akurasi dan *recall* yang cukup tinggi.
* **Naive Bayes (NB):** Menunjukkan kinerja yang paling rendah dibandingkan model lain, namun masih memiliki *recall* yang cukup baik dalam mengidentifikasi pasien berisiko.

**Kesimpulan dan Rekomendasi:**

Model **Logistic Regression**, **XGBoost**, dan **Support Vector Machine (SVM)** menunjukkan hasil yang paling menjanjikan untuk prediksi risiko stroke. Mengingat pentingnya mengidentifikasi semua pasien berisiko (tinggi *recall*), **XGBoost** dan **Logistic Regression** tampak sangat baik.

