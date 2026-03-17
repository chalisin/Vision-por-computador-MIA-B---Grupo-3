import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
 
st.title("Clasificación de imágenes")
 

#Archivo generado del entrenamiento
model = load_model('classification_model_student_background.h5')
file = st.file_uploader("Subir una imágen: ", type=["jpg", "jpeg", "png"])
 
if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
 
    if img is not None:

        img_resized = cv2.resize(img, (224, 224))
        img_final = np.expand_dims(img_resized, axis=0)
        st.image(img_resized, channels="BGR", caption="Imagen para el modelo")
        pred = model.predict(img_final)
        #se verifica el algoritmo para mostrar la predicción
        clase = "Estudiante" if pred[0][1] > pred[0][0] else "Fondo"
        st.write(f"Predicción: {clase}")
        st.write(f"Probabilidad: {pred}")
 
        
    else:
        st.error("Problemas al querer abrir analizar la imagen.")
