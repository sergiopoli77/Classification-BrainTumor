from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import google.generativeai as genai

# Ambil API Key dari environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY tidak ditemukan di file .env")
else:
    print(f"Loaded API Key: {api_key[:5]}...")

    # Debug: tampilkan model yang tersedia dan metode yang didukung
    try:
        genai.configure(api_key=api_key)
        print("Daftar model Gemini yang tersedia:")
        for model_info in genai.list_models():
            print(f"- {model_info.name} -> {model_info.supported_generation_methods}")
    except Exception as e:
        print(f"Gagal memuat daftar model Gemini: {e}")

# Fungsi untuk mengambil penjelasan penyakit
def get_disease_explanation(labels):
    if not api_key:
        return "API key untuk Gemini tidak tersedia."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')  # gunakan nama pendek, bukan 'models/...'

        joined_labels = ', '.join(labels)
        prompt = (
            f"Saya memiliki hasil analisis gambar yang mendeteksi label sebagai berikut: {joined_labels}.\n\n"
            "Berikan penjelasan singkat dan mudah dipahami oleh orang awam mengenai kemungkinan penyakit yang berkaitan dengan label-label tersebut. "
            "Sertakan:\n"
            "- Apa itu penyakit tersebut?\n"
            "- Gejala umum\n"
            "- Penyebab umum\n"
            "- Apakah berbahaya atau tidak?\n"
            "- Rekomendasi awal untuk penanganan atau konsultasi medis.\n"
            "Gunakan bahasa Indonesia yang jelas dan ringkas."
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Terjadi kesalahan saat mengakses Gemini API: {e}"

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
                prediction_result = f'Deteksi label: {", ".join(labels)}'

                explanation = get_disease_explanation(labels)

    return render_template(
        'index.html',
        result=prediction_result,
        result_image_path=result_image_path,
        explanation=explanation
    )

if __name__ == '__main__':
    app.run(debug=True)
