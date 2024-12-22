from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf 
import numpy as np
import os

app = Flask(__name__)

# Load Models
binary_model = tf.keras.models.load_model('binary_model_final.h5')
multiclass_model = tf.keras.models.load_model('multiclass_model_final.h5')

# VGG models 
VGG_binary_model = tf.keras.models.load_model('VGG_model_binary_final.h5')
VGG_multiclass_model = tf.keras.models.load_model('VGG_model_multi_final.h5')


# Prediction Function for Scratch Model
def predict_pipeline(image_path, binary_model, multiclass_model):
    img =  tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    
    binary_pred = binary_model.predict(img_array)
    if binary_pred > 0.5:
        multiclass_pred = multiclass_model.predict(img_array)
        class_idx = np.argmax(multiclass_pred)
        class_labels = {0: 'Cyst', 1: 'Stone', 2: 'Tumor'}
        return f"Abnormal: {class_labels[class_idx]}"
    else:
        return "Normal"

# Prediction Function for VGG16 Pretrained Model
def predict_pipeline_vgg(image_path, binary_model, multiclass_model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    
    binary_pred = binary_model.predict(img_array)
    if binary_pred > 0.5:
        multiclass_pred = multiclass_model.predict(img_array)
        class_idx = np.argmax(multiclass_pred)
        class_labels = {0: 'Cyst', 1: 'Stone', 2: 'Tumor'}
        return f"Abnormal: {class_labels[class_idx]}"
    else:
        return "Normal"

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload/<model_type>')
def upload(model_type):
    return render_template('index.html', model_type=model_type)

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    file = request.files['file']
    file_path = 'temp_image.jpg'
    file.save(file_path)

    if model_type == "scratch":
        result = predict_pipeline(file_path, binary_model, multiclass_model)
    elif model_type == "vgg":
        result = predict_pipeline_vgg(file_path, VGG_binary_model, VGG_multiclass_model)
    else:
        result = "Invalid Model Type"

    os.remove(file_path)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
