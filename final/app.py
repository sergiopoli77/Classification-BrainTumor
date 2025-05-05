from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Format file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Logging untuk debugging
logging.basicConfig(level=logging.INFO)

# Fungsi untuk memeriksa apakah file memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Memuat model klasifikasi (EfficientNetB1)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'effnet.keras')
try:
    model = load_model(model_path)
    logging.info("Model berhasil dimuat.")
except Exception as e:
    logging.error(f"Error saat memuat model: {e}")
    raise

# Daftar nama kelas (sesuaikan dengan data saat training)
class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Fungsi untuk memprediksi gambar yang diunggah
def predict_image(filepath):
    try:
        logging.info(f"Memproses file: {filepath}")
        # Load dan preprocessing gambar
        img = image.load_img(filepath, target_size=(150, 150))  # Ubah ukuran ke (150, 150)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisasi piksel

        # Melakukan prediksi
        predictions = model.predict(img_array)
        logging.info(f"Predictions: {predictions}")
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))

        predicted_label = class_names[predicted_class]
        logging.info(f"Predicted Label: {predicted_label}, Confidence: {confidence}")
        return predicted_label, confidence
    except Exception as e:
        logging.error(f"Error saat melakukan prediksi: {e}")
        raise

# Fungsi deskripsi hasil prediksi
def get_llm_description(predicted_label, confidence):
    descriptions = {
        'glioma_tumor': "Glioma adalah jenis tumor otak yang berasal dari sel glial.",
        'no_tumor': "Tidak ditemukan indikasi tumor pada gambar ini.",
        'meningioma_tumor': "Meningioma adalah tumor yang berasal dari meninges, lapisan pelindung otak.",
        'pituitary_tumor': "Tumor pituitari adalah tumor yang berkembang di kelenjar pituitari."
    }
    return f"<p>{descriptions.get(predicted_label, 'Informasi tidak tersedia')}<br>Tingkat keyakinan: <b>{confidence*100:.2f}%</b>.</p>"

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi gambar yang diunggah
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Only images are allowed.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(filepath)
        logging.info(f"File berhasil disimpan di: {filepath}")
    except Exception as e:
        logging.error(f"Error saat menyimpan file: {e}")
        return "Error saat menyimpan file.", 500

    try:
        predicted_label, confidence = predict_image(filepath)
        image_url = url_for('static', filename=f'uploads/{filename}')
        description = get_llm_description(predicted_label, confidence)

        return render_template('index.html',
                               prediction=predicted_label,
                               confidence=f"{confidence*100:.2f}%",
                               image_path=image_url,
                               description=description)
    except Exception as e:
        logging.error(f"Error saat memproses prediksi: {e}")
        return "Error saat memproses prediksi.", 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)