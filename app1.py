import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import base64

# Function to download the model from Google Drive
def download_model():
    gdown_link = 'https://drive.google.com/uc?id=1JJMh39JMsh8wUTdGBTL9lyN7xAIx3Vyq'  # Modified link for direct download
    gdown.download(gdown_link, 'dog_classification.h5', quiet=False)

# Load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess the image (resize, normalize, etc.)
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming the model expects 224x224 input
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Get the top 3 predicted breeds with probabilities
def get_top_3_predictions(predictions, class_names):
    top_3_indices = predictions.argsort()[0][-3:][::-1]  # Get top 3 indices
    top_3_probs = predictions[0][top_3_indices] * 100  # Convert to percentages
    top_3_breeds = [class_names[i] for i in top_3_indices]  # Get breed names
    return list(zip(top_3_breeds, top_3_probs))

# Add a background image (using CSS)
def set_background(image_file):
    # Read and encode the image file
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    # Set the background using CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_string}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

# Main function for Streamlit App
def main():
    st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")
    
    # Set the background image
    set_background('pexels-vafphotos-18126197.jpg')  # Path to your image file

    # Title and description
    st.title("🐕 Dog Breed Classifier")
    st.write("Upload a picture of a dog, and the model will classify its breed with the top 3 suggestions!")

    # Upload an image
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image with a fixed width
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)  # Reduced image size

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Download the model if it doesn't exist
        if not os.path.exists("dog_classification.h5"):
            download_model()

        # Load the model
        model = load_model("dog_classification.h5")

        # Make prediction
        predictions = model.predict(preprocessed_image)

        # Get the top 3 predictions
        class_names = [
            "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu", "Blenheim_spaniel", 
            "papillon", "toy_terrier", "Rhodesian_ridgeback", "Afghan_hound", "basset", "beagle", 
            "bloodhound", "bluetick", "black-and-tan_coonhound", "Walker_hound", "English_foxhound", 
            "redbone", "borzoi", "Irish_wolfhound", "Italian_greyhound", "whippet", "Ibizan_hound", 
            "Norwegian_elkhound", "otterhound", "Saluki", "Scottish_deerhound", "Weimaraner", 
            "Staffordshire_bullterrier", "American_Staffordshire_terrier", "Bedlington_terrier", 
            "Border_terrier", "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier", "Norwich_terrier", 
            "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier", "Sealyham_terrier", 
            "Airedale", "cairn", "Australian_terrier", "Dandie_Dinmont", "Boston_bull", 
            "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer", "Scotch_terrier", 
            "Tibetan_terrier", "silky_terrier", "soft-coated_wheaten_terrier", "West_Highland_white_terrier", 
            "Lhasa", "flat-coated_retriever", "curly-coated_retriever", "golden_retriever", 
            "Labrador_retriever", "Chesapeake_Bay_retriever", "German_short-haired_pointer", "vizsla", 
            "English_setter", "Irish_setter", "Gordon_setter", "Brittany_spaniel", "clumber", 
            "English_springer", "Welsh_springer_spaniel", "cocker_spaniel", "Sussex_spaniel", 
            "Irish_water_spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", 
            "komondor", "Old_English_sheepdog", "Shetland_sheepdog", "collie", "Border_collie", 
            "Bouvier_des_Flandres", "Rottweiler", "German_shepherd", "Doberman", "miniature_pinscher", 
            "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog", "Appenzeller", "EntleBucher", "boxer", 
            "bull_mastiff", "Tibetan_mastiff", "French_bulldog", "Great_Dane", "Saint_Bernard", 
            "Eskimo_dog", "malamute", "Siberian_husky", "affenpinscher", "basenji", "pug", "Leonberg", 
            "Newfoundland", "Great_Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond", 
            "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle", "miniature_poodle", 
            "standard_poodle", "Mexican_hairless", "dingo", "dhole", "African_hunting_dog"
        ]
        top_3 = get_top_3_predictions(predictions, class_names)

        # Display the top 3 predictions with percentages
        st.subheader("Top 3 Breed Predictions:")
        for breed, prob in top_3:
            st.write(f"**{breed}**: {prob:.2f}%")

if __name__ == "__main__":
    main()
