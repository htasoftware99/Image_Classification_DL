import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('btk.keras')

st.title('Happy or Unhappy?')

# Function to predict the class of uploaded image
def predict(image_file):
    img = image.load_img(image_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    return result[0][0]

# Main Streamlit app
def main():
    st.markdown("## Upload an image")
    file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if file is None:
        st.text("Please upload an image file.")
    else:
        image_file = file.name
        st.image(file, caption='Uploaded Image.', use_column_width=True)

        if st.button('show'):
            result = predict(file)
            if result > 0.5:
                st.write("Prediction: Positive (Happy)")
            else:
                st.write("Prediction: Negative (Unhappy)")

if __name__ == '__main__':
    main()
