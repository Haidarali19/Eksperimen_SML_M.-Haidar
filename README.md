# Eksperimen_SML_M.-Haidar
Proses Eksperimen dalam poryek pembuatan sistem machine learning memprediksi harga rumah dengan dataset House Prices. Proyek ini mencakup otomatisasi preprocessing, pelacakan eksperimen dengan MLflow, penyimpanan remote di DagsHub, dan pengemasan model menggunakan Docker.

## Struktur Repositori
- `.github/workflows/`: Berisi file konfigurasi GitHub Actions untuk CI/CD.
- `preprocessing/`: Berisi skrip otomatisasi pembersihan data (`automate_m_haidar.py`).
- `Membangun_model/`: Berisi skrip pelatihan model, pelacakan MLflow, dan bukti dokumentasi.
- `Workflow-CI/`: Berisi konfigurasi MLProject dan Docker untuk deployment model.

## Fitur Utama
- **Automated Preprocessing**: Membersihkan data secara otomatis setiap kali ada perubahan pada dataset mentah.
- **Experiment Tracking**: Menggunakan MLflow untuk mencatat parameter, metrik (MAE, R2), dan model.
- **Remote Tracking UI**: Terintegrasi dengan DagsHub untuk kolaborasi dan visualisasi eksperimen.
- **Containerization**: Model dikemas ke dalam Docker Image untuk memastikan lingkungan yang konsisten.

## Cara Menjalankan Secara Lokal

1. Instal dependensi:
   ```bash
   pip install -r Membangun_model/requirements.txt
    ```

2. Jalankan skrip preprocessing:
    python preprocessing/automate_m_haidar.py

3. Jalankan eksperimen model:
    python Membangun_model/modelling.py

4.  Final Push: 
    ```bash
    git add .
    git commit -m "Menjalankan model"
    git push origin main
    ```