

---

# Plant Disease Classification By CNN

## Overview
This project focuses on classifying different types of potato diseases using a Convolutional Neural Network (CNN). The dataset used in this project is sourced from the PlantVillage dataset on Kaggle, which includes images of potato leaves labeled as 'Potato___Early_blight,' 'Potato___Late_blight,' and 'Potato___healthy.'

## Dataset
The dataset contains 2,152 images belonging to three classes:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

The images are pre-processed and loaded into a TensorFlow dataset for efficient handling during the training and evaluation phases.

## Methodology
### 1. Data Preprocessing
   - **Image Resizing and Rescaling**: Images are resized to a uniform size of 256x256 pixels and rescaled to have pixel values in the range [0, 1]. This ensures consistency and improves model performance.
   - **Data Augmentation**: To enhance the diversity of the training dataset, data augmentation techniques such as random flipping and rotation are applied. This helps the model generalize better on unseen data.

### 2. Model Architecture
   - The model is built using a Convolutional Neural Network (CNN), which is well-suited for image classification tasks.
   - The network consists of several convolutional layers followed by max-pooling layers to capture spatial hierarchies and reduce dimensionality.
   - Finally, the output layer uses the softmax activation function to classify the input image into one of the three categories.

### 3. Training
   - The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy as the loss function.
   - The training process runs for 50 epochs, with validation on a separate dataset to monitor overfitting and model performance.

### 4. Evaluation
   - After training, the model's performance is evaluated using a test dataset that was not seen during the training phase.
   - Key metrics such as accuracy and loss are used to assess how well the model can classify potato diseases.

## Results
The model achieves good accuracy on both the validation and test datasets, indicating that it effectively differentiates between healthy and diseased potato leaves.

## Usage
To use this model:
1. Ensure you have the necessary dependencies installed, such as TensorFlow.
2. Load the dataset and preprocess it as described above.
3. Train the model using the provided architecture.
4. Evaluate the model's performance on new data to ensure it generalizes well.

## Conclusion
This project demonstrates the effectiveness of using CNNs for agricultural disease classification, which can aid in early detection and treatment, thereby reducing crop loss.

## Resources
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Introduction to Convolutional Neural Networks](https://www.youtube.com/embed/zfiSAzpy9NM)

---

