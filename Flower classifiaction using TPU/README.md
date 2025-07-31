# 🌸 Flower Classification Using TPUs

This project implements a deep learning model to classify flower species using the TensorFlow/Keras framework on Google's TPUs. Leveraging the power of TPU acceleration significantly reduces training time, allowing for faster experimentation and improved performance on image classification tasks.

# Objective

The main objective of this notebook is to:

* Build a convolutional neural network (CNN) capable of classifying images into different flower categories.

* Utilize TPUs (Tensor Processing Units) to accelerate training.

* Optimize model performance using efficient data pipelines, image augmentation, and transfer learning.

# Dataset

* The dataset used in this project is the TFFlowers Dataset, which consists of labeled images of flowers.
  
Key characteristics:

* ~3,600 total images

* Varied image sizes and lighting conditions

* Data split into training, validation, and test sets (80/10/10)

* Image Size: 512 × 512 pixels

* Classes: 103 flower categories, A full list of flower class names is available and mapped via an indexed list named CLASSES.

The dataset is stored in TFRecord format (for TPU optimization) under a cloud storage path (GCS_PATH):

* Training Files: train/*.tfrec

* Validation Files: val/*.tfrec

* Test Files: test/*.tfrec (Note: test set is unlabeled)
* 
# Environment Setup

* Framework: TensorFlow 2.x (with Keras API)

* Hardware: Google Cloud TPU via Kaggle or Colab

* Libraries Used:

    tensorflow_datasets for loading data
    
    tensorflow_addons for image augmentations
    
    matplotlib, numpy, seaborn for visualization

# Data Pipeline

 **TFRecord Decoding and Parsing**

 ➤ `decode_image(image_data)`

* Decodes JPEG byte strings.
* Casts to float and normalizes to `[0, 1]`.
* Reshapes to `(512, 512, 3)` — required for TPU compatibility.

 ➤ `read_labeled_tfrecord(example)`

* Parses labeled TFRecords with features:

  * `"image"`: image data (byte string)
  * `"class"`: integer label
* Returns: `(image, label)` pair

 ➤ `read_unlabeled_tfrecord(example)`

* Parses **test** TFRecords with features:

  * `"image"`: image data
  * `"id"`: image identifier
* Returns: `(image, id)` pair

 ➤ `load_dataset(filenames, labeled=True, ordered=False)`

* Loads TFRecord files with parallel reads.
* Ignores ordering for speed unless `ordered=True`.
* Uses `map()` to decode and parse.
* Returns:

  * Labeled: `(image, label)`
  * Unlabeled: `(image, id)`

 ➤ `data_augment(image, label)`

* Applies random horizontal flip.
* Designed to run in parallel during training for speed (thanks to `prefetch` and TPU support).


 ➤ `get_training_dataset()`

* Loads training files, applies augmentations.
* Repeats, shuffles, batches, and prefetches.

 ➤ `get_validation_dataset(ordered=False)`

* Loads validation files.
* Batches, caches, and prefetches (no augmentation or repeat).

 ➤ `get_test_dataset(ordered=False)`

* Loads test files (unlabeled).
* Batches and prefetches.

➤ `count_data_items(filenames)`

* Extracts number of images encoded in TFRecord filenames.
* Sums up across file list.

# Exploring Data

To visualize the dataset and assess model predictions, the notebook defines helper functions for displaying images and annotations:

🔹 `batch_to_numpy_images_and_labels(data)`

* Converts a TensorFlow batch into NumPy arrays.
* Handles both labeled (train/val) and unlabeled (test) datasets.
* If no labels are available (e.g., test set), returns `None` for labels.

 🔹 `title_from_label_and_target(label, correct_label)`

* Generates a descriptive title for each image:

  * Shows predicted class name.
  * Appends `[OK]` if correct, or `[NO → correct_label]` if misclassified.

 🔹 `display_one_flower(image, title, subplot, red=False, titlesize=16)`

* Displays a single flower image in a subplot with a title.
* Marks incorrect predictions in red.
* Supports dynamic title sizing and clean layout.

🔹 `display_batch_of_images(databatch, predictions=None)`

* Flexible function for batch visualization:
* Accepts `(images)`, `(images, labels)`, or `(images, labels), predictions`.
* Automatically arranges images in a square or rectangle grid.
* Supports visual comparison between predicted and actual labels.
* Adjusts layout, spacing, and title size based on image count.
* Useful for visually inspecting model output and performance.

# Model Architecture (Transfer Learning with VGG16)

| Layer Type                   | Details                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| Input                        | Shape: `(512, 512, 3)`                                         |
| Pretrained Base Model        | `VGG16` (from Keras Applications)                              |
| Weights                      | `ImageNet` pretrained weights                                  |
| `include_top`                | `False` – excludes final classification head of VGG16          |
| Trainable?                   | `No` – the base model is frozen during training                |
| Global Average Pooling Layer | Aggregates spatial features before classification              |
| Dense (Output Layer)         | Units: `103` (number of flower classes), Activation: `softmax` |


# Model Compilation

| Parameter          | Value                                     |
| ------------------ | ----------------------------------------- |
| Optimizer          | `Adam`                                    |
| Loss Function      | `SparseCategoricalCrossentropy`           |
| Evaluation Metric  | `SparseCategoricalAccuracy`               |
| Epochs             | `12`                                      |
| TPU Strategy Scope | `strategy.scope()` — for TPU acceleration |

# Training

To control how the learning rate changes during training, this project implements a custom exponential learning rate scheduler using TensorFlow's LearningRateScheduler callback.
The custom learning rate scheduler (exponential_lr) is designed to:

*Improve model convergence.

*Prevent early plateaus or unstable updates.

*Adapt the learning rate dynamically across epochs for better training efficiency.

Using a well-tuned learning rate schedule helps the model:

*Start gently (slow learning at first),

*Explore quickly during ramp-up,

*Refine efficiently during decay.

| Phase                 | Description                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Ramp-Up**           | Linearly increases learning rate from `start_lr` to `max_lr` over `rampup_epochs`. Encourages the model to "warm up" slowly without instability. |
| **Sustain**           | Keeps learning rate constant at `max_lr` for `sustain_epochs`. Useful for stable training before decay.                                          |
| **Exponential Decay** | After ramp-up/sustain, the learning rate decays exponentially toward `min_lr`, enabling fine-tuned convergence.                                  |

| Parameter        | Description                          | Value Used |
| ---------------- | ------------------------------------ | ---------- |
| `start_lr`       | Initial learning rate                | `0.00001`  |
| `max_lr`         | Maximum learning rate during ramp-up | `0.00005`  |
| `min_lr`         | Minimum learning rate after decay    | `0.00001`  |
| `rampup_epochs`  | Epochs to ramp from start to max     | `5`        |
| `sustain_epochs` | Epochs to maintain max learning rate | `0`        |
| `exp_decay`      | Decay rate (exponential factor)      | `0.8`      |

# Summary

This notebook presents a complete deep learning pipeline for classifying 103 species of flowers using high-resolution images and TPU acceleration. It begins by preprocessing TFRecord datasets, decoding images, applying augmentations, and preparing efficient TensorFlow data pipelines. The model leverages transfer learning by using a pretrained VGG16 backbone as a feature extractor, followed by a custom classification head. Training is performed within a TPU strategy scope for speed, and a custom exponential learning rate schedule is employed to improve convergence and stability. The notebook also includes comprehensive tools for visualizing batches of images, inspecting predictions, and analyzing classification performance. With well-structured code, dynamic learning rate tuning, and TPU optimization, the project demonstrates a scalable and robust approach to fine-grained image classification in TensorFlow.


