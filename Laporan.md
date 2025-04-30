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
- Menggunakan lima model machine learning yang berbeda, yaitu Random Forest Classifier, Support Vector Machine (SVM), Gaussian Naive Bayes, Logistic Regression, XGBoost Classifier
- Memilih model terbaik berdasarkan nilai F1-score, mengingat pentingnya keseimbangan antara precision dan recall dalam kasus medis.

## Data Understanding
Dataset ini dirancang untuk mendukung penelitian prediksi risiko stroke menggunakan metode machine learning dan deep learning. Dataset ini berisi 70000 baris dengan 18 kolom. Semua fitur pada dataset ini bertipe numerik, dimana gejala - gejala yang dirasakan bernilai binary (0 = Tidak merasakan, 1 = Merasakan), At Risk bernilai binary (0 = Tidak beresiko, 1 = beresiko), Stroke Risk bernilai float karena berupa persentase risiko stroke, dan Age bertipe integer berupa umur pasien. Dataset ini tidak memiliki missing value, tetapi memiliki baris duplikat sebesar 1021 baris. Terdapat outlier pada fitur Stroke Risk tetapi tidak ditangani karena hal ini normal pada medis untuk memiliki nilai persentase beresiko stroke dari rentang 5% - 100%.

Dataset disusun berdasarkan literatur medis, konsultasi dengan para ahli, serta pemodelan statistik, dan mengacu pada sumber terpercaya seperti American Stroke Association (ASA), Mayo Clinic, WHO, hingga buku teks medis seperti Harrison’s Principles of Internal Medicine.

Dataset yang digunakan dalam penelitian ini dapat diakses melalui tautan berikut: [Stroke Risk Prediction Dataset Based on Symptoms](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset).

### Variabel-variabel pada dataset:
- Chest Pain: (0 = Tidak, 1 = Ya), nyeri dada
- Shortness of Breath: (0 = Tidak, 1 = Ya), sesak napas
- Irregular Heartbeat: (0 = Tidak, 1 = Ya), detak jantung tidak teratur
- Fatigue & Weakness: (0 = Tidak, 1 = Ya), kelelahan dan kelemahan
- Dizziness: (0 = Tidak, 1 = Ya), pusing
- Swelling (Edema): (0 = Tidak, 1 = Ya), pembengkakan
- Pain in Neck/Jaw/Shoulder/Back: (0 = Tidak, 1 = Ya), nyeri di leher/rahang/bahu/punggung
- Excessive Sweating: (0 = Tidak, 1 = Ya), keringat berlebih
- Persistent Cough: (0 = Tidak, 1 = Ya), batuk kronis
- Nausea/Vomiting: (0 = Tidak, 1 = Ya), mual/muntah
- High Blood Pressure: (0 = Tidak, 1 = Ya), tekanan darah tinggi
- Chest Discomfort (Activity): (0 = Tidak, 1 = Ya), ketidaknyamanan di dada saat aktivitas
- Cold Hands/Feet: (0 = Tidak, 1 = Ya), tangan/kaki terasa dingin
- Snoring/Sleep Apnea: (0 = Tidak, 1 = Ya), mendengkur/apnea tidur
- Anxiety/Feeling of Doom: (0 = Tidak, 1 = Ya), kecemasan atau firasat buruk
- Stroke Risk (%): 0-100%, persentase estimasi risiko stroke
- At Risk: 0/1, target label (1 = berisiko, 0 = tidak berisiko)
- Age: Usia individu
Catatan:
- Fitur target untuk prediksi adalah At Risk.
- Fitur Stroke Risk (%) tidak digunakan dalam prediksi untuk menghindari data leakage.

### EDA
#### Distribusi Kolom Target (At Risk)
<img width="477" alt="DistribusiAtRisk" src="https://github.com/user-attachments/assets/46520b8b-767c-4e83-95e0-9dbc35e348b0" /></br>
Kolom "At Risk (Binary)" merupakan label klasifikasi yang menentukan apakah seseorang berisiko terkena stroke atau tidak berdasarkan berbagai faktor klinis dan gejala. Meskipun distribusi kelas menunjukkan sedikit ketidakseimbangan, namun proporsinya masih dapat diterima untuk melatih model klasifikasi tanpa perlakuan khusus balancing di tahap awal.

#### Umur Vs Stroke Risk
<img width="711" alt="StrokeRiskVsAge" src="https://github.com/user-attachments/assets/bdea77b9-c787-430a-bb15-ace936593dcb" /></br>
Dapat dilihat dari visualisasi dengan scatter plot bahwa semakin tua usia seseorang maka persentase resiko terkena stroke juga semakin tinggi. Persentase terkena stroke diatas 50% dapat dikategorikan sebagai 'At Risk' atau Beresiko Stroke.

## Data Preparation
Langkah–langkah data preparation yang dilakukan.
1. Tangani kolom duplikat
Dataset ini memiliki 1021 baris yang duplikat. Baris yang duplikat tersebut dihapus untuk meingkatkan kualitas data.
2. Feature Selection
Kolom Stroke Risk (%) di-drop karena mengandung informasi target yang seharusnya diprediksi. Kolom ini dipertahankan jika ingin melakukan tugas regresi.
3. Train-Test Split
Dataset dibagi menjadi training set sebesar 80% dan test set sebesar 20%.
4. Feature Scaling
Karena semua fitur bersifat biner (0/1) dan numerik (Age), scaling tidak wajib, tetapi dilakukan standar scaling pada Age untuk mempercepat konvergensi model Logistic Regression.

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
Lima algoritma machine learning digunakan dan dilatih pada data latih. Berikut parameter yang digunakan beserta penjelasan cara kerja masing-masing model:

   ---

   #### a. Random Forest Classifier
   - **Parameter yang Digunakan:**  
     - `n_estimators=100`  
     - `criterion='gini'` (default)
   - **Cara Kerja:**  
     Random Forest membangun banyak decision tree secara acak dari subset data dan fitur. Setiap pohon memberikan prediksi, dan hasil akhirnya ditentukan melalui mayoritas suara (voting). Teknik ini mengurangi overfitting dan meningkatkan akurasi prediksi.

   ---

   #### b. Support Vector Machine (SVC)
   - **Parameter yang Digunakan:**  
     - `kernel='rbf'` (default)
   - **Cara Kerja:**  
     SVM mencari hyperplane terbaik yang memisahkan dua kelas dengan margin maksimum. Dengan kernel RBF, data dipetakan ke ruang berdimensi lebih tinggi untuk memungkinkan pemisahan non-linear. Hanya support vectors yang memengaruhi pembentukan hyperplane.

   ---

   #### c. Gaussian Naive Bayes
   - **Parameter yang Digunakan:**  
     - Semua parameter default
   - **Cara Kerja:**  
     Naive Bayes menggunakan Teorema Bayes dengan asumsi bahwa fitur bersifat independen. Gaussian Naive Bayes mengasumsikan bahwa fitur terdistribusi normal. Probabilitas dihitung untuk tiap kelas, dan kelas dengan probabilitas tertinggi dipilih sebagai output.

   ---

   #### d. Logistic Regression
   - **Parameter yang Digunakan:**  
     - `penalty='l2'`, `solver='lbfgs'` (default)
   - **Cara Kerja:**  
     Logistic Regression menghitung kombinasi linier dari fitur dan menerapkan fungsi sigmoid untuk mengubahnya menjadi probabilitas. Model ini cocok untuk klasifikasi biner, seperti kasus prediksi risiko stroke (At Risk vs Not At Risk).

   ---

   #### e. XGBoost Classifier
   - **Parameter yang Digunakan:**  
     - `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`
   - **Cara Kerja:**  
     XGBoost adalah metode boosting yang membangun model secara bertahap. Setiap model baru memperbaiki kesalahan prediksi dari model sebelumnya menggunakan pendekatan gradien descent. XGBoost sangat kuat dan efisien dalam menangani dataset yang kompleks.

   ---
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
    * **Bagaimana Metrik Ini Bekerja:** *Accuracy* mengukur seberapa sering model membuat prediksi yang benar, baik untuk pasien yang sebenarnya berisiko stroke (TP) maupun pasien yang sebenarnya tidak berisiko (TN), dibandingkan dengan total seluruh prediksi.
    * **Interpretasi:** Semakin tinggi nilai *accuracy*, semakin baik kinerja keseluruhan model dalam mengklasifikasikan risiko stroke dengan benar. Namun, perlu diingat bahwa *accuracy* bisa menyesatkan jika dataset tidak seimbang (misalnya, jika jumlah pasien tidak berisiko jauh lebih banyak daripada yang berisiko).

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
    * **Interpretasi:** *F1-Score* yang tinggi menunjukkan bahwa model memiliki keseimbangan yang baik antara *precision* dan *recall*. 

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

## 1. Apakah Hasil Model Menjawab Setiap *Problem Statement*?

### ✅ Problem Statement 1:
**Bagaimana cara mengklasifikasikan individu berdasarkan gejala-gejala yang mereka alami menjadi 'At Risk' atau 'Not At Risk' terkena stroke?**

**Jawaban:**  
Ya. Model yang diuji — terutama Logistic Regression, XGBoost, dan SVM — mampu mengklasifikasikan individu dengan sangat baik ke dalam dua kelas: *At Risk* dan *Not At Risk*. Hal ini terbukti dari tingginya nilai F1-score dan recall, yang menunjukkan akurasi klasifikasi yang andal.

---

### ✅ Problem Statement 2:
**Seberapa akurat prediksi risiko stroke menggunakan gejala klinis sederhana tanpa memerlukan pemeriksaan medis lanjutan?**

**Jawaban:**  
Sangat akurat. Logistic Regression mencetak akurasi sempurna (1.000), disusul oleh XGBoost (0.997) dan SVM (0.990). Ini menunjukkan bahwa model dapat memprediksi risiko stroke dengan sangat tinggi hanya berdasarkan data gejala sederhana.

---

## 2. Apakah Berhasil Mencapai Setiap *Goal* yang Diharapkan?

### ✅ Goal 1:
**Membuat model machine learning untuk memprediksi risiko stroke (At Risk: 0/1) berdasarkan gejala-gejala yang dilaporkan.**

**Status:**  
Tercapai. Model telah dibangun dan dievaluasi dengan data gejala sederhana. Beberapa model digunakan untuk membandingkan performa.

---

### ✅ Goal 2:
**Menilai performa model berdasarkan metrik klasifikasi seperti accuracy, precision, recall, dan F1-score.**

**Status:**  
Tercapai. Semua metrik dilaporkan lengkap untuk setiap model, dan hasil evaluasi menunjukkan bahwa beberapa model mencapai skor sangat tinggi.

---

## 3. Apakah Setiap *Solution Statement* Berdampak dan Efektif?

### 🔹 Solution 1:
**Menggunakan lima model machine learning berbeda:**
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Logistic Regression
- XGBoost Classifier

**Evaluasi Dampak:**  
- Penggunaan berbagai model memberikan variasi perspektif dan validasi silang performa.
- Logistic Regression menunjukkan performa sempurna (semua metrik = 1.000), sangat andal untuk klasifikasi biner sederhana.
- XGBoost dan SVM juga menunjukkan performa sangat tinggi dan seimbang pada semua metrik.
- Random Forest dan Naive Bayes tetap memberikan hasil yang baik dan relevan untuk analisis perbandingan.
- Pendekatan ini terbukti efektif dalam mengidentifikasi model terbaik secara objektif berdasarkan data evaluasi.

---

### 🔹 Solution 2:
**Memilih model terbaik berdasarkan nilai F1-score, mengingat pentingnya keseimbangan antara precision dan recall.**

**Evaluasi Dampak:**  
- F1-score terbukti efektif sebagai acuan pemilihan model.
- Model seperti XGBoost dan Logistic Regression memiliki F1-score mendekati sempurna, memastikan klasifikasi yang andal dan aman.

---

## Kesimpulan

Hasil evaluasi model **sepenuhnya mendukung Business Understanding**:

- ✅ Semua *problem statement* terjawab dengan baik.
- 🎯 *Goals* tercapai secara penuh, didukung dengan metrik evaluasi lengkap.
- 🚀 *Solusi* yang diterapkan terbukti efektif dan berdampak positif, terutama dalam konteks medis.

Model seperti **Logistic Regression** dan **XGBoost** sangat cocok digunakan sebagai alat bantu skrining awal risiko stroke berdasarkan gejala sederhana. Ini memberi potensi besar untuk diterapkan pada layanan kesehatan primer atau aplikasi masyarakat berbasis teknologi.

