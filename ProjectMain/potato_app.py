import streamlit as st
import numpy as np
from PIL import Image
from skimage import transform
from joblib import load

# Load the trained SVM model
model_path = "potato_model.pkl"
svm_model = load(model_path)

# Function to preprocess images
def preprocess_img(img, target_features=16384):
    # Resize image to the required dimensions (128*128)
    img_resized = transform.resize(img, (128, 128), anti_aliasing=True, preserve_range=True)

    # Convert to RGB if the image has a single channel
    if len(img_resized.shape) == 2:
        img_rgb = np.stack([img_resized] * 3, axis=-1)
    else:
        img_rgb = img_resized

    # Reshape the image to (128, 128, 3)
    img_reshaped = np.reshape(img_rgb, (128, 128, 3))

    # Flatten the image
    img_flattened = img_reshaped.flatten()

    # Reshape the flattened image to the target number of features
    img_output = np.reshape(img_flattened, (-1, target_features))

    return img_output

# Function to make predictions
# def predict_image(img):
#     img_preprocessed = preprocess_img(img)
#     img_preprocessed_flat = img_preprocessed.reshape(1, -1)
#     prediction = svm_model.predict(img_preprocessed_flat)
#     return prediction[0]
def predict_image(img):
    img_preprocessed = preprocess_img(img)
    img_preprocessed_flat = img_preprocessed.reshape(1, -1)

    # Check the shape of the input data
    print("Input shape:", img_preprocessed_flat.shape)

    prediction = svm_model.predict(img_preprocessed_flat)
    return prediction[0]

# Streamlit app
st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🍃")
st.title("🍃 Potato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png", ".jfif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button('Predict'):
        image_resized = np.array(image)  # Convert PIL Image to NumPy array
        prediction = predict_image(image_resized)

        if prediction == 'Potato___Late_blight':
            st.error("⚠️ Late blight detected in the leaf!")
    
        else:
            st.success("✅ No disease detected in the leaf.")

st.markdown("---")
st.markdown("ONE PR")