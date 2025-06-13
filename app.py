from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_cnn_model.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image = request.files["digit"]
        img = Image.open(image).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        pred = model.predict(img_array)
        prediction = np.argmax(pred)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
