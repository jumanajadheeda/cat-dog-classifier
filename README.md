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

Loaded using:

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


Epochs: 10

🚀 How to Use the Model
1️⃣ Load the Model
from tensorflow.keras.models import load_model
model = load_model("model.keras")

2️⃣ Predict on a New Image
import tensorflow as tf

img = tf.keras.utils.load_img("your_image.jpg", target_size=(256, 256))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Dog")
else:
    print("Cat")

📈 Training Highlights

Automatic image loading & batching

Normalization for faster convergence

Dropout to reduce overfitting

Validation accuracy improves steadily

Works well even with a small model

📌 Requirements

Install dependencies:

pip install tensorflow matplotlib numpy

📝 Notes

Supports .keras and .h5 model formats

Use 256×256 resolution images for best prediction

🤝 Contributing

Pull requests are welcome!
Feel free to submit improvements or suggestions.

👩‍💻 Author

Jumana
B.Tech CSE | ML & AI Enthusiast
