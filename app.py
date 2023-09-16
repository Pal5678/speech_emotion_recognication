import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model

# Function to extract MFCC features for prediction
def extract_mfcc_for_prediction(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
def generate_spectrogram_image(y, sr, predicted_emotion):
    plt.figure(figsize=(8, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram ({predicted_emotion})')
    spectrogram_filename = f'spectrogram_{predicted_emotion}.png'  # Include emotion label in the filename
    plt.savefig(spectrogram_filename)
    plt.close()
    return spectrogram_filename
# Load your Keras model
model = load_model('model.h5')  # Replace with your model path

# Define Streamlit user interface elements
st.title('Emotion Detection App')
st.write('Upload an audio file and get emotion prediction.')

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Preprocess the uploaded audio file
    audio_data, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
    
    # Extract MFCC features for prediction
    mfcc_features = extract_mfcc_for_prediction(audio_data, sr)

    # Prepare the data for prediction
    X_pred = np.expand_dims(mfcc_features, axis=0)
    X_pred = np.expand_dims(X_pred, axis=-1)

    # Make predictions using the loaded Keras model
    predictions = model.predict(X_pred)

    # Convert the predictions to emotion labels
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_emotion_index]

    # Display the prediction
    st.write(f'Predicted Emotion: {predicted_emotion}')
    
    # Add an audio player for the uploaded file
    st.audio(audio_data, format='audio/wav', sample_rate=sr)


    # Display the audio waveform
    st.subheader('Audio Waveform')
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr, color='b')
    st.pyplot()

    # Display the prediction and confidence score
    st.write(f'Predicted Emotion: {predicted_emotion}')
    st.write(f'Confidence Score: {predictions[0][predicted_emotion_index]:.2f}')

    # Visualize emotion probabilities as a bar chart
    st.subheader('Emotion Probabilities')
    fig,ax = plt.subplots()
    ax.bar(class_labels, predictions[0])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

