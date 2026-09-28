from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR =Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

# AGE_MODEL_PATH = MODELS_DIR / "Age_model.keras"
# GENDER_MODEL_PATH = MODELS_DIR / "Gender_model.keras"

import tensorflow as tf

AGE_MODEL_PATH = MODELS_DIR / "age_saved_model"
GENDER_MODEL_PATH = MODELS_DIR / "gender_saved_model"

print("Loading age model...")
age_model = tf.saved_model.load(AGE_MODEL_PATH)

print("Loading gender model...")
gender_model = tf.saved_model.load(GENDER_MODEL_PATH)

gender_dict = {
    0: "Male",
    1: "Female",
}



#Image Processing Function
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image into the format expected by the models.

    Original pipeline:

    image
    -> resize 64*64
    -> grayscale
    -> numpy array
    -> reshape (1, 64, 64, 1)
    -> normalize [0, 1]
    """
    # Resize the image to 64x64
    image = image.resize((64, 64))

    # Convert to grayscale
    image = image.convert("RGB")

    # Convert to numpy array
    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0

    image_array = image_array.reshape(1, 64, 64, 3)

    return image_array.astype(np.float32)


#prediction function

def predict_age_gender(image: Image.Image) -> dict:
    """
    Run age and gender prediction on one image.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict age
    age_prediction = age_model.predict(preprocessed_image,verbose=0)
    predicted_age=int(np.round(age_prediction[1])[0][0]) # Assuming the model outputs a single value for age

    # Predict gender
    gender_prediction = gender_model.predict(preprocessed_image,verbose=0)

    pre_gender = np.array(gender_prediction)  # Convert to numpy array if it's a list
    # Process the gender prediction
    predicted_gender_class = (pre_gender >= 0.5).astype(int)[0]  # Get binary class (0 or 1)
    predicted_gender = gender_dict[predicted_gender_class[0][0]]
    

    return {
        "age": predicted_age,
        "gender": predicted_gender,
        "confidence": float(gender_prediction[0][0])
    }