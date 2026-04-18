from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# ==================== CONFIG ====================
MODEL_WEIGHTS_PATH = 'cat_dog_model.h5'
IMG_SIZE = 128
# ===============================================

def create_model():
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

print("🔄 Loading model architecture and weights...")

try:
    model = create_model()
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("✅ Model weights successfully loaded!")
except Exception as e:
    print("❌ Model load failed!", str(e))
    exit()

# ====================== ROUTES ======================

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)
        prob_dog = float(prediction[0][0])

        if prob_dog > 0.5:
            result_class = "dog"
            confidence = prob_dog
        else:
            result_class = "cat"
            confidence = 1 - prob_dog

        return jsonify({
            "class": result_class,
            "confidence": round(confidence, 4),
            "probability_dog": round(prob_dog, 4)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Server started on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)