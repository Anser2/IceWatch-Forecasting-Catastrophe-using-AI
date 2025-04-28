import gdown
import os
import rasterio
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
import os

# Define MODEL_PATH using absolute path
MODEL_PATH = os.path.join(os.getcwd(), "models/torrentflow/CHILL")
FILE_ID = '1GEG8lmja5M6R7d_YSsG6s3zZJlYqTC6j'  # Google Drive file ID
DOWNLOAD_NAME = 'latest_image.tif'
IMAGE_PATH = f'models/torrentflow/tiff_imagery_for_cnn/{DOWNLOAD_NAME}'

# --- Image Loading Function (Added from your working script) ---
def load_multiband_image(path, target_size=(128, 128)):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # shape: (bands, H, W)
    img = np.transpose(img, (1, 2, 0))       # shape: (H, W, bands)
    img = tf.image.resize(img, target_size)
    img = img / 255.0  # normalization (crucial for model compatibility)
    return img

# --- Load model ---
def load_model():
    try:
        return tf.saved_model.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# --- Download image ---
def download_image(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# --- Predict ---
def load_and_predict(image_path):
    # Load and preprocess image (now using proper function)
    input_image = load_multiband_image(image_path)
    input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension

    # Load model and make prediction
    model = load_model()
    predict_fn = model.signatures["serving_default"]
    prediction = predict_fn(input_image)

    # Extract output - adjust key if needed (check model signatures)
    output_key = list(prediction.keys())[0]
    result = prediction[output_key].numpy()[0][0]  # [batch][class]
    return result

# # --- Main ---
# if __name__ == '__main__':
#     print("📥 Downloading latest image...")
#     download_image(FILE_ID, IMAGE_PATH)

#     print("🤖 Predicting GLOF...")
#     try:
#         prediction = load_and_predict(IMAGE_PATH)
#     except Exception as e:
#         print(f"❌ Prediction failed: {str(e)}")
#         exit(1)

#     # Formatting prediction for readability
#     confidence = f"{prediction:.2%}"
#     if prediction >= 0.5:
#         print(f"🚨 GLOF likely detected! Confidence: {confidence}")
#     else:
#         print(f"✅ No GLOF detected. Confidence: {confidence}")