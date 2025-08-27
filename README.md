# human_action_recognition

### **Optimizing Convolutional Neural Networks for Classifying Human Activities from Images**



**Project Overview -** This project focuses on the classification of 15 different human actions from still images using Convolutional Neural Networks (CNNs). The goal is to explore both a simple baseline CNN model and advanced pre-trained architectures, evaluating their performance in terms of accuracy, generalization, and computational efficiency.



We compare a basic CNN against four state-of-the-art deep learning models:

\- ResNet50: Uses residual connections to overcome gradient vanishing in deep networks.

\- InceptionV3: Learns features at multiple scales using its inception modules.

\- Xception: Uses depthwise separable convolutions for efficiency without sacrificing accuracy.

\- MobileNet: Lightweight architecture optimized for low computational cost and real-time applications.



**Dataset:**



The dataset used consists of images of human actions, split into training and testing sets:



Training set: Training\_set.csv with image file paths and labels.

Testing set: Testing\_set.csv with image file paths and labels.

Train Folder: Folder with corresponding training images.

Test Folder: Folder with corresponding test images.



The dataset contains 15 different human activity labels.





**Dependencies**:

* Python 3.x
* TensorFlow 2.x / Keras
* OpenCV (cv2)
* NumPy
* Pandas
* Matplotlib
* Seaborn
* PIL





**Code Summary:**

Data Exploration \& Visualization

* Count distribution of labels
* Display sample images for each class



Data Preprocessing

* Resizing images to 128x128 pixels
* Normalization of pixel values
* One-hot encoding of labels
* Model Development



Baseline CNN: Three convolutional layers with pooling and dropout.

Transfer Learning Models: ResNet50, InceptionV3, Xception, MobileNet with frozen pre-trained weights, fine-tuned using custom dense layers.



Training \& Evaluation

* Train/validation split (75%/25%)
* Model training for 10 epochs
* Loss and accuracy visualization
* Random sample predictions with actual vs predicted labels



Performance Analysis

* Baseline CNN shows overfitting.
* MobileNet achieves the best balance between accuracy and computational efficiency.





**Usage:**

* Clone the repository.
* Ensure the dataset is placed in the correct directory structure.
* Run the notebook or Python script to train models and evaluate their performance.
* Visualizations of predictions and training metrics will be displayed automatically.





**Results:**

* Baseline CNN achieves decent training accuracy but overfits on validation data.
* Transfer learning models improve generalization significantly.
* MobileNet provides the best trade-off between accuracy, model size, and inference speed.



**Future Work:**

* Implement data augmentation to further reduce overfitting.
* Fine-tune the deeper layers of pre-trained networks.
* Explore additional architectures to improve accuracy.
