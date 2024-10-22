import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

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

# Get the label of the predicted breed
def get_prediction_label(prediction):
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
    
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

# Main function for Streamlit App
def main():
    st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="wide")

    # Add a background image (using CSS)
    def set_background(image_file):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url({image_file});
                background-size: cover;
            }}
            </style>
            """, unsafe_allow_html=True
        )
    
    set_background('pexels-pixabay-531880.jpg')  # Your background image file

    st.title("🐕 Dog Breed Classifier")
    st.write("Upload a picture of a dog, and the model will classify its breed!")

    # Upload an image
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Load the model
        model = load_model("dog_classification.h5")

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_breed = get_prediction_label(prediction)

        # Display the prediction
        st.success(f"The predicted breed is: **{predicted_breed}** 🐾")

if __name__ == "__main__":
    main()
