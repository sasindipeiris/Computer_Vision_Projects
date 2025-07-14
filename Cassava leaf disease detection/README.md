# Overview

Cassava is a vital food crop in Africa, particularly for smallholder farmers. Unfortunately, cassava plants are susceptible to viral diseases which significantly reduce crop yields. The current diagnosis process involves field experts inspecting plants manually—a process that is both slow and resource-intensive.

To address this, the competition provides 21,367 labeled images of cassava leaves collected from farms in Uganda. Most were crowdsourced using mobile-quality cameras and later annotated by experts. The task is to build a machine learning model that accurately classifies images into one of five disease categories, under constraints mimicking real-world use (low bandwidth, mobile images, limited hardware).

This project focuses on building a baseline image classification model to identify cassava leaf diseases using TensorFlow, TPUs, and transfer learning. It was developed in the context of the Cassava Leaf Disease Classification competition on Kaggle.

# Libraries used

| Library             | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `tensorflow`        | Model building and TPU training                     |
| `keras`             | High-level API for model layers and compilation     |
| `numpy`             | Numerical computations                              |
| `pandas`            | Tabular data manipulation                           |
| `matplotlib.pyplot` | Visualization                                       |
| `KaggleDatasets`    | For accessing Kaggle-hosted datasets                |
| `sklearn`           | Dataset splitting and evaluation                    |
| `functools.partial` | For configuring functions with pre-filled arguments |

# Dataset

The dataset includes:

  Training images of cassava leaves
  
  A CSV with image IDs and corresponding disease labels

The diseases include:

  Cassava Bacterial Blight (CBB)
  
  Cassava Brown Streak Disease (CBSD)
  
  Cassava Mosaic Disease (CMD)
  
  Healthy Leaves
  
  Unknown category (to be predicted)

# Preprocessing

In the preprocessing stage of the cassava leaf disease detection pipeline, the dataset is read and prepared from TFRecord files, which are a TensorFlow-specific format optimized for performance. Depending on whether the data is labeled (for training and validation) or unlabeled (for testing), different schemas are applied while parsing the records. Labeled examples contain both image data and corresponding disease class labels, while test records include image data along with unique image names. Each image is decoded from JPEG format into a normalized tensor with pixel values scaled between 0 and 1, ensuring consistency in input to the model. The images are also reshaped to a fixed size to match the model’s input requirements. To improve model generalization, a simple data augmentation strategy is applied by randomly flipping images horizontally, which enhances variability in the dataset. Additionally, the input pipeline uses efficient techniques like prefetching, which allows the data loading process to be overlapped with model training, particularly beneficial when running on TPUs. The training and validation data is split using an 65-35 ratio to ensure a robust evaluation of model performance. This well-structured and optimized preprocessing setup enables faster and more reliable training of deep learning models for image classification.

# Model 


* **Distributed Training Scope**: The model is built within a `strategy.scope()` block, which enables distributed training across multiple devices such as TPUs or GPUs, ensuring scalability and faster computation.

* **Input Preprocessing**: A `Lambda` layer applies the preprocessing function specific to ResNet50 (`preprocess_input`), which normalizes the input images appropriately for the pre-trained model.

* **Base Model (Feature Extractor)**:

  * The backbone of the model is **ResNet50**, a deep convolutional neural network pre-trained on ImageNet.
  * It is used with `include_top=False`, meaning the final classification layers of ResNet50 are removed.
  * The base model’s weights are **frozen** (`trainable=False`) to retain pre-trained knowledge and avoid overfitting during initial training.

* **Batch Normalization**:

  * A `BatchNormalization` layer with renormalization (`renorm=True`) is placed before the model to stabilize learning by normalizing the inputs.

* **Custom Classification Head**:

  * A `GlobalAveragePooling2D` layer reduces the spatial dimensions of the feature maps, converting them into a single vector per image.
  * This is followed by a `Dense` layer with 8 units and ReLU activation to introduce non-linearity and learn task-specific patterns.
  * A final `Dense` layer with the number of output neurons equal to the number of cassava classes is added, using a **softmax** activation function for multi-class classification.

* **Compilation Settings**:

  * The model is compiled with the **Adam optimizer**, using a learning rate scheduler and an epsilon of 0.001 for numerical stability.
  * The **loss function** is `sparse_categorical_crossentropy`, suitable for categorical classification when labels are provided as integers.
  * **Metrics** used include `sparse_categorical_accuracy` to measure classification accuracy directly on the label format.

This model design effectively combines the power of transfer learning with a lightweight custom classifier, enabling efficient training and high performance on the cassava leaf disease classification task.






