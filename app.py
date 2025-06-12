import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for, jsonify

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_DIR = 'models'

print("--- Memuat model Keras... ---")
try:
    models = {
        'utama': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model_utama.h5')),
        'cnn': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model_cnn.h5')),
        'vgg16': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model_vgg16.h5'))
    }
    model_names = {
        'utama': 'Model Utama (ResNet)',
        'cnn': 'CNN Kustom',
        'vgg16': 'VGG16'
    }
    print("--- Model berhasil dimuat ---")
except Exception as e:
    print(f"--- Error saat memuat model: {e} ---")
    models = {}
    model_names = {}

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'File tidak valid'})

    if file:
        selected_model_key = request.form.get('model')
        if selected_model_key not in models:
            return jsonify({'error': f'Model {selected_model_key} tidak ditemukan'})

        upload_folder = os.path.join('static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        processed_image = preprocess_image(file_path)
        
        model = models[selected_model_key]
        prediction = model.predict(processed_image)

        result_text = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        
        # Kembalikan hasil dalam format JSON
        return jsonify({
            'prediction_text': result_text,
            'image_url': image_url,
            'model_display_name': model_names.get(selected_model_key, '')
        })

if __name__ == '__main__':
    app.run(debug=True)