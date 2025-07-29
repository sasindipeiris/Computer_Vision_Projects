# Overview

This notebook presents a comprehensive machine learning pipeline to solve the Higgs Boson classification challenge — identifying whether a given particle interaction detected by the ATLAS experiment at CERN corresponds to a Higgs boson event or not. The dataset originates from a Kaggle competition hosted by CERN and ATLAS.

Large and complicated datasets like these are where deep learning excels. In this notebook, we'll build a Wide and Deep neural network to determine whether an observed particle collision produced a Higgs boson or not.


---

# Libraries used

| **Library**                      | **Purpose / Description**                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `tensorflow`                     | Main deep learning framework; used for building and training neural networks.                          |
| `tensorflow.distribute`          | Handles distributed training across multiple devices like TPUs/GPUs.                                   |
| `tensorflow.keras`               | High-level API for building models, layers, callbacks, and training logic.                             |
| `pandas`                         | Data manipulation and analysis, especially for loading and exploring datasets.                         |
| `matplotlib.pyplot`              | Used for plotting and visualizing data and model results.                                              |
| `kaggle_datasets.KaggleDatasets` | Enables access to datasets hosted on Kaggle, especially useful in notebooks running on Kaggle kernels. |
| `tensorflow.io.FixedLenFeature`  | Used to define features with fixed-length for reading TFRecord files.                                  |
| `tf.data.experimental.AUTOTUNE`  | Optimizes performance of data pipelines by automatically tuning the parallelism.                       |

---

# TPU Detection

In this notebook, **TPU (Tensor Processing Unit)** usage is integrated to accelerate deep learning model training. TPUs are specialized hardware developed by Google, optimized for performing high-throughput, low-latency matrix operations, which are common in deep learning workloads.

The notebook intelligently detects the presence of a TPU and, if available, initializes it using TensorFlow’s TPU strategy. This enables the model to be trained in a distributed fashion across multiple TPU cores, significantly reducing training time, especially on large datasets. If a TPU is not available, it gracefully falls back to using a single GPU or CPU.

By leveraging TPUs, the notebook ensures efficient resource utilization and enhanced performance, making it well-suited for computationally intensive tasks such as particle classification in the Higgs Boson challenge.

---

# Dataset loading

The Higgs dataset contains 21 "low-level" features of the decay products and also 7 more "high-level" features derived from these.

The dataset has been encoded in a binary file format called TFRecords. These two functions will parse the TFRecords and build a TensorFlow tf.data.Dataset object that we can use for training.

These two functions are part of a **data input pipeline** for TensorFlow models, designed to load and decode data stored in the **TFRecord** format — a highly efficient binary format commonly used for large-scale machine learning datasets.



 **`make_decoder` – Feature Parser Factory**

The `make_decoder` function **creates and returns a decoder function** that can parse each individual record in a TFRecord file.

🔹 What It Does:

* Accepts a `feature_description` dictionary that describes the structure of each example (e.g., what keys exist and what types they are).
* The returned decoder:

  * Parses the serialized `example` into structured data using that description.
  * Extracts the `'features'` field (likely a serialized tensor), decodes it, and reshapes it into the correct format (28-element float vector in this case).
  * Extracts the `'label'` field (classification target).
  * Returns the `features` and `label` as a pair ready for model training or inference.

🧩 Why It’s Important:

* It **decouples** TFRecord decoding logic from dataset loading, making it reusable and modular.
* Ensures correct formatting and type casting of data before feeding it into a model.


**`load_dataset` – TFRecord Loader with Preprocessing**

The `load_dataset` function loads one or more TFRecord files into a TensorFlow `tf.data.Dataset` object using the decoder from `make_decoder`.

🔹 What It Does:

* Takes a list of TFRecord file paths and a decoder function.
* Reads the data in parallel using `AUTO` tuning for efficiency.
* Optionally **shuffles or randomizes** the data reading order for training (if `ordered=False`).
* Applies the decoder to every example using `.map()` — converting serialized bytes into usable `(features, label)` pairs.

 ⚙️ Optimization:

* It uses TensorFlow’s `AUTOTUNE` feature to automatically optimize data loading performance.
* Setting `ordered=False` disables deterministic order to **speed up training** by allowing for asynchronous, non-ordered reads (helpful in large datasets).



 🧾 Overall Use Case

These functions are typically used together in a TensorFlow training pipeline:

1. **Define the feature schema** (`feature_description`).
2. **Create a decoder** using `make_decoder`.
3. **Load the dataset** using `load_dataset`, apply the decoder, and return a fully prepared `tf.data.Dataset`.
4. Feed the dataset into a model for **training**, **validation**, or **inference**.

---

# Feature Description Function

📌 Purpose:

* Defines the **structure** of each example stored in a TFRecord.
* Acts like a **blueprint** telling TensorFlow:

  * What keys to expect in each record
  * What type and shape each feature should have



 🧱 Structure:


  feature_description = {
      'features': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.float32),
  }


| **Key**      | **Type**                          | **Explanation**                                                                                              |
| ------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `'features'` | `FixedLenFeature([], tf.string)`  | Serialized tensor (e.g., 28 floats) stored as a string. Requires decoding with `tf.io.parse_tensor()` later. |
| `'label'`    | `FixedLenFeature([], tf.float32)` | A single floating-point value representing the class label.                                                  |



 🔍 What is `FixedLenFeature`?

* Tells TensorFlow to expect a **fixed-size, single-value** input.
* Syntax: `FixedLenFeature(shape, dtype)`

  * `shape=[]` → a scalar
  * `dtype=tf.string` or `tf.float32`, etc.



 🔧 Why Use `tf.string` for Features?

* In some datasets, especially when saving arrays/tensors, the feature vector is **pre-serialized** into a byte string for storage efficiency.
* Later, you decode it using `tf.io.parse_tensor()` during parsing.


 💡 When to Customize This:

* If your TFRecord contains images, audio, or multi-dimensional tensors, the `feature_description` will change accordingly.
* For categorical data or text, you may use `tf.int64`, `tf.string`, or even `VarLenFeature`.

---

dataset_size = 11 million (total number of examples).

validation_size = 500,000 reserved for validation.

training_size = 10.5 million used for training.

batch_size is scaled by the number of TPU/GPU replicas (strategy.num_replicas_in_sync) to maximize parallelism.

steps_per_epoch and validation_steps are calculated by dividing dataset size by batch size.

steps_per_execution = 256 improves TPU/GPU performance by reducing Python interaction frequency.

| **Step**                | **Training Dataset (`ds_train`)**                   | **Validation Dataset (`ds_valid`)**                 | **Reasoning / Purpose**                                                                                        |
| ----------------------- | --------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Load TFRecords**      | `load_dataset(train_files, decoder, ordered=False)` | `load_dataset(valid_files, decoder, ordered=False)` | Loads serialized TFRecords and applies the decoder to return `(features, label)` pairs.                        |
| **.cache()**            | ✅ Yes                                               | ✅ Yes                                               | Caches data in memory or local storage to speed up multiple epochs or evaluations.                             |
| **.repeat()**           | ✅ Yes                                               | ❌ No                                                | Training data is repeated indefinitely to provide continuous batches across epochs. Not needed for validation. |
| **.shuffle(2**19)\*\*   | ✅ Yes (large buffer)                                | ❌ No                                                | Adds randomness to training input order to avoid overfitting or learning input order patterns.                 |
| **.batch(batch\_size)** | ✅ Yes                                               | ✅ Yes                                               | Groups examples into mini-batches for efficient training and evaluation.                                       |
| **.prefetch(AUTO)**     | ✅ Yes                                               | ✅ Yes                                               | Overlaps data preparation and model execution, improving training speed and hardware utilization.              |

---

# Wide and Deep Model Architecture

| **Component**                   | **Description / Code**                                                                | **Reasoning / Purpose**                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Function: `dense_block`**     | Builds a repeated unit consisting of: <br> `Dense → BatchNorm → Activation → Dropout` | Modularizes the building block for the deep neural network. Promotes reusability and clean architecture. |
| `layers.Dense(units)`           | Fully connected layer with `units` neurons                                            | Learns feature interactions. More layers → deeper representation.                                        |
| `layers.BatchNormalization()`   | Normalizes activations                                                                | Stabilizes and accelerates training by reducing internal covariate shift.                                |
| `layers.Activation(activation)` | Applies chosen non-linearity (e.g., ReLU)                                             | Introduces non-linearity to help model complex patterns.                                                 |
| `layers.Dropout(dropout_rate)`  | Randomly disables neurons during training                                             | Prevents overfitting by promoting generalization.                                                        |


| **Block**                      | **Deep Model (DNN)**                 | **Reasoning**                                       |
| ------------------------------ | ------------------------------------ | --------------------------------------------------- |
| `keras.Input(shape=[28])`      | Input layer expecting 28 features    | Matches the reshaped feature vector from TFRecords. |
| 5 × `dense_block(...)`         | Stack of 5 hidden layers (deep path) | Enables hierarchical feature learning.              |
| `layers.Dense(1)`              | Final output layer with 1 unit       | Outputs a single logit for binary classification.   |
| `keras.Model(inputs, outputs)` | Constructs the deep model            | Encapsulates the entire deep sub-network.           |


| **Block**                          | **Wide Model (`LinearModel`)**  | **Reasoning**                                                                        |
| ---------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| `keras.experimental.LinearModel()` | A linear model for memorization | Captures simple rules and feature combinations directly, complements the deep model. |


| **Block**                               | **Wide & Deep Fusion**                           | **Reasoning**                                                                                                                        |
| --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `keras.experimental.WideDeepModel(...)` | Combines `LinearModel` (wide) and `Model` (deep) | Merges memorization (wide) and generalization (deep) capabilities. Proven effective in structured data (e.g., Kaggle tabular tasks). |
| `activation='sigmoid'`                  | Applies sigmoid to final output                  | Suitable for binary classification, producing output in range (0, 1).                                                                |


| **Block**                               | **Wide & Deep Fusion**                           | **Reasoning**                                                                                                                        |
| --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `keras.experimental.WideDeepModel(...)` | Combines `LinearModel` (wide) and `Model` (deep) | Merges memorization (wide) and generalization (deep) capabilities. Proven effective in structured data (e.g., Kaggle tabular tasks). |
| `activation='sigmoid'`                  | Applies sigmoid to final output                  | Suitable for binary classification, producing output in range (0, 1).                                                                |

---

# Training

In this model training setup, two key callbacks are used to improve efficiency and generalization: **EarlyStopping** and **ReduceLROnPlateau**.

The **EarlyStopping** callback is configured with a `patience` of 2 and a `min_delta` of 0.001. This means the training will stop if the validation loss does not improve by at least 0.001 for two consecutive epochs. This prevents unnecessary training once the model stops making meaningful progress, saving computational resources and reducing the risk of overfitting. Additionally, `restore_best_weights=True` ensures that the model reverts to the weights from the epoch with the best validation performance, not the final epoch, leading to better generalization on unseen data.

The **ReduceLROnPlateau** callback monitors the model's performance and reduces the learning rate by a factor of 0.2 when no improvement is observed (`patience=0`). This adaptive adjustment allows the optimizer to make finer updates when progress stalls, helping the model converge more precisely toward an optimal solution. The learning rate will not go below `min_lr=0.001`, ensuring it doesn't become too small to learn effectively.

Together, these callbacks form a smart training control system—**early stopping avoids overtraining**, while **adaptive learning rate scaling enhances convergence and model refinement**.

