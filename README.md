🐶🐱 Cat vs Dog Image Classifier

A deep learning-based image classifier that distinguishes between cats and dogs using TensorFlow + Keras.
The model is trained on an 8,000-image dataset and achieves strong performance with a simple CNN architecture.

📁 Project Structure
cat-dog-classifier/
│
├── model.keras              
├── cat-dog-classifier.h5    
├── README.md                 
└── (scripts / notebooks)

📦 Dataset
Total images: 8,000
Training: 6,400
Validation: 1,600

Classes:
🐱 Cat
🐶 Dog

Dataset loaded using:
image_dataset_from_directory(
    "cat_vs_dog/dataset",
    image_size=(256, 256),
    batch_size=32
)

🧠 Model Summary
Type: Convolutional Neural Network (CNN)
Layers: Conv → MaxPool → Dropout → Dense
Optimizer: Adam
Loss: Binary Crossentropy
Training Epochs: 10

How to Use the Model
1️ Load the Model
from tensorflow.keras.models import load_model
model = load_model("model.keras")

2️ Predict on a New Image
import tensorflow as tf

img = tf.keras.utils.load_img("your_image.jpg", target_size=(256, 256))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Dog")
else:
    print("Cat")

Training Highlights
Automatic image loading & batching
Normalization layer for better convergence
Dropout to prevent overfitting
Validation accuracy improves consistently
Works well even with small model size

Requirements
Install required libraries:
pip install tensorflow matplotlib numpy

Notes
Works with .keras and .h5 models.
Use images 256×256 for prediction.

Contributing
Pull requests are welcome!
If you want to improve the model or add features, feel free to open an issue.

Author
Jumana
B.Tech CSE | ML & AI Enthusiast