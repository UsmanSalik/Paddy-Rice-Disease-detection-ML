Below is a clean, professional, GitHub-ready README.md written exactly according to your CIFAR-10 CNN notebook, including training, augmentation, visualization, and testing.

You can copy–paste this directly into README.md and push it to GitHub.

🧠 CIFAR-10 Image Classification using CNN (TensorFlow)

This project implements a deep Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow & Keras.
The model is trained with data augmentation, batch normalization, dropout, L2 regularization, and includes feature map visualization to understand how the CNN learns visual patterns.

📌 Dataset

CIFAR-10 consists of 60,000 color images (32×32) in 10 classes:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

Split used in this project:

Training: 45,000 images

Validation: 5,000 images

Testing: 10,000 images

🏗️ Model Architecture

The CNN architecture is designed for deep feature extraction and generalization:

🔹 Convolution Blocks

Conv2D → BatchNormalization → ReLU

Filter progression: 32 → 64 → 128 → 256

Kernel size: 3×3

Padding: same

L2 Regularization (weight decay = 0.0001)

🔹 Regularization

Dropout layers: 0.2 → 0.3 → 0.4 → 0.5

Batch Normalization after every convolution

🔹 Pooling

MaxPooling2D (2×2) after each convolution block

🔹 Output

Flatten layer

Dense layer with Softmax activation for 10 classes

🔄 Data Preprocessing

Convert images to float32

Normalize using mean and standard deviation of training set

One-hot encoding for labels

X = (X - mean) / (std + 1e-8)

🔁 Data Augmentation

Applied using ImageDataGenerator to improve generalization:

Rotation

Width & height shifting

Horizontal flip

Zoom

Brightness adjustment

Shear

Channel shifting

⚙️ Training Configuration

Optimizer: Adam

Learning Rate: 0.0005

Loss: Categorical Crossentropy

Batch Size: 64

Epochs: 50

📉 Callbacks

ReduceLROnPlateau – reduces LR when validation loss plateaus

EarlyStopping – prevents overfitting and restores best weights

📊 Training Visualization

The project plots:

Training vs Validation Loss

Training vs Validation Accuracy

These graphs help analyze:

Overfitting

Convergence behavior

Model stability

🔍 Feature Map Visualization

To understand how the CNN learns, feature maps are extracted from each Conv2D layer:

A new model is created to output intermediate convolution layers

Feature maps are visualized using Matplotlib

Helps analyze edge detection, texture learning, and object patterns

🧪 Model Evaluation

The trained model is evaluated on the test dataset:

Test Accuracy

Test Loss

This gives an unbiased performance measure.

🌐 External Image Testing

The model pipeline also supports loading and preprocessing external images from URLs using:

urllib

OpenCV

RGB conversion

This demonstrates real-world usability beyond CIFAR-10.

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

OpenCV

Scikit-Learn

Jupyter Notebook

🚀 How to Run
git clone https://github.com/UsmanSalik/cifar10_cnn_visualization.git
cd cifar10_cnn_visualization
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Open the notebook:

jupyter notebook

📌 Project Highlights

✔ Deep CNN with regularization
✔ Data augmentation for robustness
✔ Feature map visualization
✔ Clean training workflow
✔ Real-world image inference

👤 Author

Usman Salik
AI / Deep Learning Enthusiast
Focused on Computer Vision & Neural Networks
