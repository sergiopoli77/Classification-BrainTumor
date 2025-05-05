from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model .keras
model_path = os.path.join(os.path.dirname(__file__), 'model', 'effnet.keras')
model = tf.keras.models.load_model(model_path)

# Daftar nama kelas
class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Fungsi untuk prediksi gambar
def predict_image(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((150, 150))  # sesuai input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index]

    return predicted_label, confidence

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)
    image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html',
                           prediction=label,
                           confidence=f"{confidence*100:.2f}%",
                           image_path=image_url)
                           
# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
