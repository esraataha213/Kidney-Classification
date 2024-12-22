# CT Kidney Disease Classification

Welcome to the CT Kidney Disease Classification project! This repository focuses on detecting and classifying abnormalities in CT kidney images. Two different approaches are used: models trained from scratch and VGG16 pretrained models. Both pipelines are designed to predict whether a kidney is normal or abnormal and, if abnormal, classify the type of abnormality.

## Dataset Overview
Our dataset contains 12,446 unique samples distributed as follows:

- *Normal*: 5,077 images
- *Cyst*: 3,709 images
- *Stone*: 1,377 images
- *Tumor*: 2,283 images

## Models

### Scratch Models
- *Binary Model*: Classifies images as Normal or Abnormal.
- *Multiclass Model*: Classifies Abnormal images into one of the three classes: Cyst, Stone, or Tumor.

### VGG16 Pretrained Models
- *VGG Binary Model*: Fine-tuned binary classifier using VGG16.
- *VGG Multiclass Model*: Fine-tuned multiclass classifier using VGG16.

## Prediction Pipelines

### Scratch Model Prediction
python
binary_model = tf.keras.models.load_model('binary_model_final.h5')
multiclass_model = tf.keras.models.load_model('multiclass_model_final.h5')

def predict_pipeline(image_path, binary_model, multiclass_model):
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


### VGG16 Model Prediction
python
VGG_binary_model = tf.keras.models.load_model('VGG_model_binary_final.h5')
VGG_multiclass_model = tf.keras.models.load_model('VGG_model_multi_final.h5')

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


## Usage

1. *Prepare Models:*
   - Ensure the models (binary_model_final.h5, multiclass_model_final.h5, VGG_model_binary_final.h5, VGG_model_multi_final.h5) are saved in your working directory.

2. *Run Prediction:*
   - Use the predict_pipeline function for scratch models.
   - Use the predict_pipeline_vgg function for VGG16 models.

3. *Example:*
python
result = predict_pipeline('path_to_image.jpg', binary_model, multiclass_model)
print(result)


## License
This project is open-source under the MIT License. Feel free to contribute and improve!
