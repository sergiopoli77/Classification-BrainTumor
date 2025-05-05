from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi model
model = None
model_path = os.path.join('model', 'best.pt')

# Cek apakah file model ada
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

    if request.method == 'POST':
        file = request.files.get('image')

        if file:
            # Simpan gambar upload
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # Baca gambar dengan OpenCV
            img = cv2.imread(image_path)
            if img is None:
                prediction_result = 'Error: Could not read the image.'
            elif model is None:
                prediction_result = 'Error: Model is not loaded properly.'
            else:
                # Prediksi dengan YOLO
                results = model.predict(source=img, imgsz=640, conf=0.5)
                result_img = results[0].plot()

                # Simpan hasil gambar
                result_filename = 'result.jpg'
                result_image_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_image_path, result_img)

                prediction_result = 'Prediction done. See result below.'

    return render_template('index.html', result=prediction_result, result_image_path=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
