# pip install gtts
from gtts import gTTS
import streamlit as st
import numpy as np
import pickle
import os
from pyngrok import ngrok
from googletrans import Translator
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.title('Emotion-Aware Bi-lingual Conversational Assistant with Speech Output')


with open('/content/model.pickle', 'rb') as f:
    emotion_model = pickle.load(f)

with open('/content/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


label_mapping = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}


def final_prediction_model(text):
  sentence=tokenizer.texts_to_sequences([text])
  padded_sentence=pad_sequences(sentence,maxlen=100,padding='pre')
  prediction=emotion_model.predict(padded_sentence)
  predicted_label=label_mapping[np.argmax(prediction)]
  return predicted_label


def text_to_speech(text,name="input"):
    audio_file = f"{name}_text.wav"
    tts = gTTS(text=text, lang='en')
    tts.save(audio_file)
    return audio_file



def translate_text(text, source_lang='en', target_lang='hi'):
    translator = Translator()
    translated = translator.translate(text, src=source_lang, dest=target_lang)
    st.write(f"Original text : {text}")
    return translated.text


user_input = st.text_input("Enter text to detect emotion and generate audio:")


if user_input:

    input_audio_file=text_to_speech(user_input)
    st.audio(input_audio_file, format="audio/wav")

    translated_input_text = translate_text(user_input)
    st.write(f"Translated_text : {translated_input_text}")

    predicted_label = final_prediction_model(user_input)
    st.write(f"Predicted Emotion : {predicted_label}")


    output_audio_file = text_to_speech(predicted_label,predicted_label)
    st.audio(output_audio_file, format="audio/wav")

    translated_output_text=translate_text(predicted_label)
    st.write(f"Translated_emotion_text : {translated_output_text}")
