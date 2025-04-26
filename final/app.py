from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import requests
import markdown
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mendapatkan API key dari file .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY tidak ditemukan di file .env")
else:
    print(f"Loaded API Key: {api_key[:5]}...")  # Menampilkan sebagian API key untuk keamanan

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Format file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Fungsi untuk memeriksa apakah file yang diunggah memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Memuat model YOLOv8
model_path = os.path.join(os.path.dirname(__file__), 'model', 'brain_tumor_best.pt')
model = YOLO(model_path)

# Daftar nama kelas untuk prediksi
class_names = ['Normal', 'Tumor']

# Fungsi untuk memprediksi gambar yang diunggah
def predict_image(filepath):
    # Melakukan prediksi menggunakan YOLOv8
    results = model(filepath)
    predictions = results[0].boxes.data  # Mendapatkan hasil prediksi

    # Mengambil kelas dengan confidence tertinggi
    if len(predictions) > 0:
        predicted_class = int(predictions[0][5].item())  # Indeks kelas
        confidence = float(predictions[0][4].item())  # Confidence score
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = "Tidak ada deteksi"
        confidence = 0.0

    # Mengembalikan label prediksi dan tingkat keyakinan
    return predicted_label, confidence

# Fungsi untuk mendapatkan deskripsi dari Gemini API berdasarkan hasil prediksi
def get_llm_description(predicted_label, confidence):
    try:
        # URL endpoint API Gemini
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Header untuk permintaan API
        headers = {
            "Content-Type": "application/json"
        }

        # Membuat prompt untuk dikirim ke Gemini API
        prompt = (
            f"Hasil prediksi menunjukkan kondisi: '{predicted_label}' dengan tingkat keyakinan {confidence*100:.2f}%. "
            f"Berikan analisis mendalam tentang kondisi ini, termasuk:\n\n"
            f"1. Penjelasan umum tentang kondisi '{predicted_label}'.\n"
            f"2. Risiko atau dampak jika kondisi ini tidak ditangani.\n"
            f"3. Saran awal untuk langkah-langkah yang dapat diambil.\n"
            f"4. Kapan pengguna harus segera memeriksakan diri ke dokter.\n\n"
            f"Selalu tekankan bahwa ini hanya informasi awal, konsultasi dokter sangat disarankan."
        )

        # Payload untuk permintaan API
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        # Mengirim permintaan POST ke API
        response = requests.post(api_url, json=payload, headers=headers)
        print("FULL JSON RESPONSE:", response.json())  # Debugging respons API

        # Memproses respons API
        if response.status_code == 200:
            candidates = response.json().get("candidates", [{}])
            description = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            html_description = markdown.markdown(description)  # Mengubah teks menjadi HTML
            return html_description
        else:
            return f"<p>Error: {response.status_code} - {response.text}</p>"
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return "<p>Terjadi kesalahan saat meminta penjelasan dari Gemini.</p>"

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi gambar yang diunggah
@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah file diunggah
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Memeriksa apakah file memiliki format yang diizinkan
    if not allowed_file(file.filename):
        return "Invalid file type. Only images are allowed.", 400

    # Menyimpan file yang diunggah
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Melakukan prediksi dan mendapatkan deskripsi
    predicted_label, confidence = predict_image(filepath)
    image_url = url_for('static', filename=f'uploads/{filename}')
    description = get_llm_description(predicted_label, confidence)

    # Mengirim data ke template HTML
    return render_template('index.html',
                           prediction=predicted_label,
                           confidence=f"{confidence*100:.2f}%",
                           image_path=image_url,
                           description=description)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)