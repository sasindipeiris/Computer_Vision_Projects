# Overview

This project involves building a machine learning pipeline to detect key landmarks on human faces — such as eyes, eyebrows, nose, and mouth corners — using the Kaggle Facial Keypoints Detection dataset.

# Dataset

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. There are 15 keypoints, which represent the following elements of the face:

left_eye_center, right_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_inner_corner, right_eye_outer_corner, left_eyebrow_inner_end, left_eyebrow_outer_end, right_eyebrow_inner_end, right_eyebrow_outer_end, nose_tip, mouth_left_corner, mouth_right_corner, mouth_center_top_lip, mouth_center_bottom_lip

Left and right here refers to the point of view of the subject.

In some examples, some of the target keypoint positions are misssing (encoded as missing entries in the csv, i.e., with nothing between two commas).

The input image is given in the last field of the data files, and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels.

Data files :

* training.csv: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.
  
* test.csv: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels
  
* submissionFileFormat.csv: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location. FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. Location is what you need to predict.

* # Libraries used

* | **Import Statement**                                                            | **Module/Library**        | **Purpose**                                                              |
| ------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------ |
| `import zipfile`                                                                | `zipfile` (Python stdlib) | Extracts `.zip` files (used to load the dataset archives).               |
| `import os`                                                                     | `os` (Python stdlib)      | Handles file paths and directory operations.                             |
| `import pandas as pd`                                                           | `pandas`                  | Loads and manipulates CSV data for keypoints and images.                 |
| `import numpy as np`                                                            | `numpy`                   | Performs numerical operations, array manipulation, and image formatting. |
| `import matplotlib.pyplot as plt`                                               | `matplotlib`              | Visualizes images and predicted keypoints for evaluation.                |
| `from xgboost import XGBRegressor`                                              | `XGBoost`                 | Predicts missing keypoints using gradient boosting regression.           |
| `from sklearn.model_selection import train_test_split`                          | `scikit-learn`            | Splits dataset into training and validation sets.                        |
| `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score` | `scikit-learn`            | Evaluates regression model performance.                                  |
| `from tensorflow import keras`                                                  | `TensorFlow`              | Provides deep learning framework for building CNNs.                      |
| `from keras import layers`                                                      | `Keras` (via TensorFlow)  | Defines convolutional and dense layers for the neural network.           |

# Data Preprocessing

| **Step**                                      | **Code**                                                                                                                                                                                            | **Explanation / Reasoning**                                                                                                                                                                                                                                                                     |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Define Paths to Zip Files**              | `python<br>zip_train = '../input/facial-keypoints-detection/training.zip'<br>zip_test = '../input/facial-keypoints-detection/test.zip'<br>extract_to_path = '../working/'<br>`                      | Specifies paths for the **training and test zip files** and the **directory** to which they'll be extracted. This is necessary before loading or parsing the dataset.                                                                                                                           |
| **2. Extract Training and Test Data**         | `python<br>with zipfile.ZipFile(zip_train, 'r') as zip_ref:<br> zip_ref.extractall(extract_to_path)<br>with zipfile.ZipFile(zip_test, 'r') as zip_ref:<br> zip_ref.extractall(extract_to_path)<br>` | Opens each ZIP file and **extracts its contents** (e.g., CSV or image files) into a working directory. This step makes the data accessible for loading.                                                                                                                                         |
| **3. Initialize Image Container**             | `python<br>all_image = []<br>`                                                                                                                                                                      | Creates an empty list to hold **all parsed images** as arrays of integers.                                                                                                                                                                                                                      |
| **4. Convert Image Strings to Integer Lists** | `python<br>for all_pixel in data['Image'].values:<br> image = list(map(int, all_pixel.split(' ')))<br> all_image.append(image)<br>`                                                                 | - Each entry in `data['Image']` is a **flattened image stored as a space-separated string**.<br>- `split(' ')` splits the string into individual pixel values.<br>- `map(int, ...)` converts strings to integers.<br>- Appends the image as a list of pixel intensities (0–255) to `all_image`. |
| **5. Convert to NumPy Array and Reshape**     | `python<br>all_image = np.array(all_image).reshape(-1, 96, 96)<br>`                                                                                                                                 | - Converts the list of lists into a **NumPy array** for efficient numerical computation.<br>- Reshapes each 1D array of length 96×96 = 9216 into a **2D image of shape (96, 96)**.<br>- `-1` infers the number of images based on total size.                                                   |
| **6. Remove Incomplete Samples**              | `python<br>data_new = data.dropna()<br>data_new.info()<br>`                                                                                                                                         | Drops all rows with **missing values** (NaNs) to ensure only **complete keypoint annotations** are used for training. `info()` confirms the shape and completeness of the cleaned dataset.                                                                                                      |

# Using XGBoost to predict the missing values


This project aims to predict missing facial keypoints from grayscale facial images using XGBoost, fill in incomplete data, and visualize predictions before passing the data to a deep learning model.

🔹 1. Feature and Target Selection

The dataset contains 30 facial keypoints (each with x and y coordinates), some of which may be missing. We choose a subset of the most consistently labeled keypoints as the **features** (inputs) and the rest as **targets** (outputs). This allows the model to learn how the positions of some keypoints can predict the positions of others.

---

🔹 2. Splitting the Dataset

Once features and targets are defined, the dataset is split into training and testing sets. This is essential to evaluate the model's ability to generalize to unseen data. A small portion (10%) of the data is reserved as the test set, while the remaining data is used for training the XGBoost model.

---

 🔹 3. Evaluation Metrics for Regression

To measure the performance of the trained model, we use three regression metrics:

* **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
* **Mean Squared Error (MSE)**: Emphasizes larger errors by squaring the differences, useful for penalizing bad predictions.
* **R² Score**: Indicates how well the model explains the variance in the target data. An R² score closer to 1 means better performance.

These metrics help determine whether the model has learned meaningful relationships between known and missing keypoints.

---

🔹 4. Training the XGBoost Regressor

XGBoost is chosen due to its robustness, speed, and effectiveness with structured/tabular data. It learns how to map known keypoints (like eye centers or mouth corners) to the positions of missing ones using gradient-boosted decision trees.

---

 🔹 5. Predicting Missing Keypoints

Once trained, the model is used to predict missing keypoints in the dataset. This step is crucial because the dataset originally contains rows with partial labels. Filling these gaps using model predictions ensures we can utilize the entire dataset in the next phase of training (e.g., deep learning).

---

 🔹 6. Visualizing Predictions

To qualitatively assess how well the model performs, we overlay the original image with both true and predicted keypoints. Ground-truth points (from the original dataset) and predicted points (from the model) are plotted on the same face image. This gives an intuitive view of how accurate and reliable the model’s predictions are.

Such visualizations help spot systematic errors (e.g., the model always misplacing the nose) and guide further improvements.

---

🔹 7. Filling Missing Values in the Dataset

After generating predictions, these are used to replace any `NaN` values in the original dataset. This results in a **complete dataset** with no missing keypoints, enabling us to use all samples instead of discarding incomplete ones.

Completing the dataset in this way preserves data diversity and maximizes training effectiveness in later stages.

---

 🔹 8. Reshaping and Preparing for CNN Training

All facial images are reshaped from flat arrays into their original 96×96 grayscale format. This reshaped data, along with the now-complete keypoint coordinates, is then split again into training and testing sets. The resulting dataset is fully preprocessed and ready to be used as input for a Convolutional Neural Network (CNN) model.

CNNs can now be trained end-to-end using the entire dataset, leveraging the spatial structure of the image along with complete keypoint labels.

---

# Modeling - CNN



This model is a subclass of `keras.Model` and is designed to take grayscale facial images of shape **(96×96×1)** and output **30 facial keypoint coordinates** (15 points × x and y).



 🔹 1. Convolutional Blocks (`block_conv2d`)

Each block is responsible for extracting increasingly abstract features from the input image using:

🔸 Conv2D Layers

* Each block contains **two 2D convolutional layers** that learn spatial filters to detect patterns like edges, textures, and parts of facial features.
* Kernel size = 3: Captures local details while maintaining resolution.
* Padding is implicitly 'valid' (default), meaning output size reduces slightly.

🔸 Batch Normalization

* Applied after each convolution to **normalize activations**, helping speed up training and improve generalization.
* Reduces internal covariate shift.

 🔸 Leaky ReLU Activations

* Chosen instead of regular ReLU to prevent "dying neurons" — Leaky ReLU allows a small gradient when inputs are negative.

 🔸 MaxPooling2D

* Each block ends with a 2×2 pooling layer to **downsample** the spatial dimensions by a factor of 2.
* This reduces computational load and introduces translation invariance.
---
 🔹 2. Block Progression (conv1 → conv4)

Each block doubles the number of filters (channels), allowing the network to learn from **low-level to high-level** representations:

| Block | Filters   | Output Shape (Approx.) | Purpose                      |
| ----- | --------- | ---------------------- | ---------------------------- |
| conv1 | 16 → 16   | 94×94 → 92×92 → 46×46  | Capture low-level features   |
| conv2 | 32 → 32   | 44×44 → 42×42 → 21×21  | Extract mid-level patterns   |
| conv3 | 64 → 64   | 19×19 → 17×17 → 8×8    | Learn part structures        |
| conv4 | 128 → 128 | 6×6 → 4×4 → 2×2        | Capture high-level semantics |

> Each block reduces image resolution but increases feature depth, capturing progressively more abstract facial patterns.
---
🔹 3. Fully Connected Layers (`self.fc`)

After the final convolutional output is flattened, it's passed through a fully connected head to predict keypoints:

🔸 Flatten

* Converts the 2×2×128 output tensor into a 1D vector of size 512 (2×2×128) for dense processing.

🔸 Dense(128) + Leaky ReLU

* First dense layer compresses the feature vector to 128 dimensions.
* Leaky ReLU ensures gradient flow during training.

🔸 Dense(256) + Leaky ReLU

* Expands to a higher dimension, enabling richer combinations of features before final regression.

🔸 Dense(30)

* Final output layer with 30 units for **15 (x, y) facial keypoint pairs**.
* No activation function here — regression output (continuous coordinates).

---

🔹 4. `call` Method (Forward Pass)

The `call()` method defines how data flows through the model:

* Input → conv1 → conv2 → conv3 → conv4 → Flatten → FC layers → Output

This design allows TensorFlow to trace the model for training and inference.

---

 🔹 5. Summary and Input Shape

* The model is instantiated and initialized with a dummy input of shape (32, 96, 96, 1), which simulates a batch of 32 grayscale images.
* `model.summary()` prints the total number of parameters and architecture, helping you track model complexity.

---

 ✅ Why This Architecture Works for Keypoint Detection

* **Progressive convolutional blocks** extract increasingly complex facial features.
* **Batch normalization and Leaky ReLU** improve training stability and prevent vanishing gradients.
* **Dense layers** at the end interpret the feature vector to regress precise keypoint locations.
* **Output of 30 neurons** directly maps to the expected coordinates, avoiding classification and working in a regression setting.

# CNN Training Configuration

| **Component**        | **Parameter / Setting**  | **Description and Reasoning**                                                                |
| -------------------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| **Optimizer**        | `Adam`                   | Adaptive learning rate optimizer, ideal for complex models with sparse gradients.            |
| **Learning Rate**    | `0.001`                  | Initial step size for weight updates; a standard starting point for Adam.                    |
| **Loss Function**    | `LogCosh`                | Smooth, robust loss that behaves like MSE for small errors but like MAE for large ones.      |
| **Metric**           | `LogCoshError`           | Monitors how well predictions match targets during training/validation.                      |
| **Batch Size**       | `32`                     | Number of images processed before weight update; balances memory efficiency and convergence. |
| **Epochs**           | `50`                     | Maximum number of training iterations over the full dataset.                                 |
| **Validation Split** | `0.1`                    | 10% of training data used for validation to monitor overfitting.                             |
| **Steps per Epoch**  | `x_train.shape[0] // 32` | Total number of training steps in each epoch.                                                |


# Evaluation

🔹 **Training and Validation Loss Trends (LogCosh)**

From the loss curve:

* **Sharp Initial Drop (Epochs 1–5):**
  The model rapidly learns core patterns during the first few epochs, reducing both training and validation loss from very high values to around 1.5. This indicates effective early learning.

* **Stabilization (Epochs 10–20):**
  After the steep descent, both losses converge and start to flatten around a value close to 1. This suggests the model has captured the majority of the learnable patterns.

* **Plateau and Smooth Convergence (Epochs 20–45):**
  Both training and validation losses remain low and relatively parallel, which is a strong sign of:

  * **Good generalization** (no major overfitting or underfitting),
  * **Stable training**, and
  * **Well-tuned learning rate scheduler.**

---

🔹 **Evaluation Metrics**

On the test set:

* **MAE (Mean Absolute Error): 1.32**
  On average, the predicted keypoint coordinates are only \~1.3 pixels off from the true values on a 96×96 image — a solid result for keypoint detection.

* **MSE (Mean Squared Error): 3.69**
  The squared error is low, suggesting no major outliers or extreme mispredictions.

* **R² Score: 0.72**
  The model explains \~72% of the variance in the facial keypoints, indicating **strong predictive power**, especially considering the input is grayscale imagery.



