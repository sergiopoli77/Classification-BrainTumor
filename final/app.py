from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import requests
import markdown
import google.generativeai as genai

# Ambil API Key dari environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY tidak ditemukan di file .env")
else:
    print(f"Loaded API Key: {api_key[:5]}...")


# Konfigurasi Flask
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi model YOLOv8
model = None
model_path = os.path.join('model', 'best.pt')

if not os.path.exists(model_path):
    print(f"Error: The model file at {model_path} does not exist.")
else:
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


# Fungsi untuk mengambil penjelasan penyakit
def get_llm_description(labels):
    try:
        # URL endpoint API Gemini
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Header untuk permintaan API
        headers = {
            "Content-Type": "application/json"
        }

        # Membuat prompt untuk dikirim ke Gemini API
        prompt = (
            f"Hasil prediksi menunjukkan kondisi: '{labels}'  "
            f"Berikan analisis mendalam tentang kondisi ini, termasuk:\n\n"
            f"1. Penjelasan umum tentang kondisi '{labels}'.\n"
            f"2. Risiko atau dampak jika kondisi ini tidak ditangani.\n"
            f"3. Saran awal untuk langkah-langkah yang dapat diambil mengobati atau mencegah Glaukoma jika.\n"
            f"4. Kapan pengguna harus segera memeriksakan diri ke dokter mata.\n\n"
            f"Selalu tekankan bahwa ini hanya informasi awal, konsultasi dokter sangat disarankan untuk dilakukan"
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
            explanation = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            html_description = markdown.markdown(explanation)  # Mengubah teks menjadi HTML
            return html_description
        else:
            return f"<p>Error: {response.status_code} - {response.text}</p>"
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return "<p>Terjadi kesalahan saat meminta penjelasan dari Gemini.</p>"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    result_image_path = None
    explanation = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            if img is None:
                prediction_result = 'Error: Gagal membaca gambar.'
            elif model is None:
                prediction_result = 'Error: Model tidak dimuat dengan benar.'
            else:
                results = model.predict(source=img, imgsz=640, conf=0.5)
                result_img = results[0].plot()

                result_filename = 'result.jpg'
                result_image_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_image_path, result_img)

                labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
                if labels:
                    prediction_result = f'Deteksi label: {", ".join(labels)}'
                    explanation = get_llm_description(", ".join(labels))
                else:
                    prediction_result = "Tidak terdeteksi adanya penyakit pada gambar."
                    explanation = "<p>Hasil deteksi menunjukkan tidak ada tanda-tanda penyakit yang teridentifikasi oleh tumor. Otak kemungkinan dalam kondisi <strong>normal</strong> atau tidak menunjukkan anomali visual yang dapat dideteksi secara otomatis.</p>"

    return render_template(
        'index.html',
        result=prediction_result,
        result_image_path=result_image_path,
        explanation=explanation
    )

if __name__ == '__main__':
    app.run(debug=True)
