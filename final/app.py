from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

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

        # Validasi indeks kelas
        if 0 <= predicted_class < len(class_names):
            predicted_label = class_names[predicted_class]
        else:
            predicted_label = "Indeks kelas tidak valid"
            confidence = 0.0
    else:
        predicted_label = "Tidak ada deteksi"
        confidence = 0.0

    # Mengembalikan label prediksi dan tingkat keyakinan
    return predicted_label, confidence

# Fungsi placeholder untuk deskripsi (tidak menggunakan Gemini API)
def get_llm_description(predicted_label, confidence):
    return f"<p>Hasil prediksi menunjukkan kondisi: '{predicted_label}' dengan tingkat keyakinan {confidence*100:.2f}%.</p>"

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