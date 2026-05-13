from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

# Percobaan 10 Metriks
REQUEST_COUNT = Counter('ml_model_requests_total', 'Total jumlah request ke model')
LATENCY = Summary('ml_model_inference_latency_seconds', 'Waktu yang dibutuhkan untuk prediksi')
MODEL_MAE = Gauge('ml_model_mae_current', 'Mean Absolute Error saat ini')
MODEL_R2 = Gauge('ml_model_r2_score', 'R2 Score saat ini')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU')
MEM_USAGE = Gauge('system_memory_usage_bytes', 'Penggunaan Memori')
PREDICTION_VALUE = Gauge('ml_model_last_prediction_value', 'Nilai prediksi terakhir')
ERROR_COUNT = Counter('ml_model_errors_total', 'Total error saat inferensi')
ACTIVE_USERS = Gauge('ml_model_active_users', 'Jumlah user aktif saat ini')
DATA_DRIFT_SCORE = Gauge('ml_model_data_drift_score', 'Skor deteksi data drift')

def process_request():
    """Simulasi proses monitoring"""
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    # Simulasi perhitungan
    time.sleep(random.uniform(0.1, 0.5)) 
    LATENCY.observe(time.time() - start_time)
    
    # Update metriks lainnya
    MODEL_MAE.set(random.uniform(200000, 210000))
    MODEL_R2.set(random.uniform(0.02, 0.05))
    CPU_USAGE.set(random.uniform(10, 80))
    MEM_USAGE.set(random.uniform(1024**2, 1024**3))
    PREDICTION_VALUE.set(random.uniform(100000, 900000))
    ACTIVE_USERS.set(random.randint(1, 50))
    DATA_DRIFT_SCORE.set(random.uniform(0, 1))
    
    if random.random() < 0.05:
        ERROR_COUNT.inc()

if __name__ == '__main__':
    # Jalankan server metrics di port 8000
    start_http_server(8000)
    print("Prometheus Exporter berjalan di port 8000...")
    while True:
        process_request()
        time.sleep(1)