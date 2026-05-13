import pandas as pd
import joblib
import os

# Load Model
model_path = r"D:\Projek\SMSL_M. Haidar\Eksperimen_SML_M.-Haidar\Membangun_model\model.pkl"

def run_inference():
    print("Memulai Proses Inferensi")
    try:
        if not os.path.exists(model_path):
            print(f"Error: File tidak ditemukan di {model_path}")
            return

        # Memuat file .pkl menggunakan joblib
        model = joblib.load(model_path)
        print("Berhasil: Model .pkl berhasil dimuat.")

        # Data input dummy untuk inferensi
        data_dummy = pd.DataFrame([[3, 2, 1500, 2000, 1, 0, 0, 3, 1200, 300, 2010, 0]], 
                                 columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                                          'floors', 'waterfront', 'view', 'condition',
                                          'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'])

        # Menghitung Prediksi
        prediction = model.predict(data_dummy)
        print("-" * 35)
        print(f"Hasil Prediksi Harga Rumah: ${prediction[0]:,.2f}")
        print("-" * 35)

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_inference()