from flask import Flask, request, jsonify, render_template, session, url_for
import os
import secrets
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# try importing keras load_model only if available
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except Exception as e:
    load_model = None
    image = None

app = Flask(__name__)
app.secret_key = "secretkey"

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATHS = ["plant_disease_model.h5", os.path.join("model","plant_disease_model.h5")]

# create folders if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# try to load model from known locations, but don't crash if missing
model = None
if load_model:
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = load_model(p)
                print(f"[INFO] Loaded model from: {p}")
                break
            except Exception as e:
                print(f"[WARN] Failed loading model at {p}: {e}")

if model is None:
    print("[WARN] No model found or failed to load. App will run in demo mode (returns a placeholder prediction).")

class_names = ["Healthy","Leaf Spot","Rust","Powdery Mildew"]

def predict_image(filepath):
    # If model not loaded, return a demo label
    if model is None or image is None:
        return "Demo - Model not loaded (Healthy)"

    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    return result

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}),400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error":"No file selected"}),400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        prediction = predict_image(filepath)

        static_filename = "upload_" + secrets.token_hex(8) + ".jpg"
        static_path = os.path.join(app.config["STATIC_FOLDER"], static_filename)

        # save a copy to static so templates can serve it
        Image.open(filepath).save(static_path)

        session["prediction"] = prediction
        # store the static filename only; result.html must use url_for to build path
        session["image_path"] = static_filename

        # cleanup temp upload
        os.remove(filepath)

        return jsonify({"success":True})
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error":str(e)}),500

@app.route("/result")
def result():
    prediction = session.get("prediction")
    image_path = session.get("image_path")
    # send to template; template will call url_for('static', filename=image_path)
    return render_template("result.html",
                           prediction=prediction,
                           image_path=image_path)

if __name__ == "__main__":
    # For quick demo use, disable debug in production
    app.run(debug=True)