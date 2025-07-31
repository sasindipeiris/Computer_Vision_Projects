# Overview

This notebook demonstrates a modern and efficient feature extraction and matching pipeline for computer vision tasks using the following components:

* DINOv2 (Self-Supervised Vision Transformer) for dense feature extraction

* Alike (A Light-weight Learned Keypoint Detector) for sparse keypoint detection and description

* LightGlue (Lightweight Learned Feature Matcher) for robust, geometry-aware keypoint matching

Together, these components create a powerful and fast system for image matching, localization, SLAM, and visual correspondence tasks.

# Objective

The goal of this project is to:

* Evaluate the effectiveness of combining pretrained DINO features with Alike keypoints.

* Use LightGlue to perform advanced matching with geometric consistency and attention-based filtering.

* Provide a clean and efficient demonstration of the full pipeline from image loading to visualization.

* # Libraries used

| Library / Module      | Component                         | Description / Purpose                                                                                         |
|-----------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| `time`                | `time`, `sleep`                   | Used for measuring execution time and controlling flow (e.g., timed waits).                                   |
| `gc`                  | —                                 | Python garbage collector interface for memory management.                                                     |
| `numpy`               | `np`                              | Numerical operations on arrays, used for tensors, reshaping, etc.                                             |
| `h5py`                | —                                 | Reads and writes HDF5 files, often used for storing keypoints, descriptors, and other large data efficiently. |
| `dataclasses`         | —                                 | Simplifies the creation of data container classes with automatic `__init__`, `__repr__`, etc.                 |
| `pandas`              | `pd`                              | Data analysis and manipulation library, useful for tabular logging and display.                               |
| `IPython.display`     | `clear_output`                    | Clears notebook output during loops or updates for cleaner display.                                           |
| `collections`         | `defaultdict`                     | Dictionary subclass that provides default values for missing keys.                                            |
| `copy`                | `deepcopy`                        | Creates a full (deep) copy of an object to avoid modifying the original data.                                 |
| `PIL`                 | `Image`                           | Python Imaging Library (Pillow) for opening and manipulating images.                                          |
| `cv2`                 | —                                 | OpenCV: used for image processing tasks like resizing, drawing, keypoint visualization.                       |
| `torch`               | —                                 | PyTorch deep learning library used for tensor operations and model inference.                                 |
| `torch.nn.functional` | `F`                               | Functional API in PyTorch for activations, loss functions, and tensor ops (e.g., `F.relu`).                   |
| `kornia`              | `K`                               | Differentiable computer vision operations in PyTorch (e.g., color space transforms, geometric transforms).    |
| `kornia.feature`      | `KF`                              | Kornia’s module for feature detection and description (e.g., SIFT, SuperPoint).                               |
| `lightglue`           | `match_pair`                      | Matches keypoints between image pairs using LightGlue.                                                        |
| `lightglue`           | `ALIKED`, `LightGlue`             | ALIKED: lightweight keypoint detector & descriptor. LightGlue: learned keypoint matcher.                      |
| `lightglue.utils`     | `load_image`, `rbd`               | Utilities for image loading and processing (exact use of `rbd` may vary).                                     |
| `transformers`        | `AutoImageProcessor`, `AutoModel` | Hugging Face Transformers API for loading pretrained image models and processors (e.g., DINO).                |
| `pycolmap`            | —                                 | Python bindings for COLMAP, used for 3D reconstruction and structure-from-motion tasks.                       |
| `sys`                 | `sys.path.append()`               | Adds custom script paths (e.g., for Kaggle competitions or utility functions).                                |
| `database`            | `*`                               | Custom module for managing COLMAP database interactions.                                                      |
| `h5_to_db`            | `*`                               | Converts `.h5` feature files into COLMAP-compatible SQLite database format.                                   |
| `metric`              | —                                 | Custom or competition-provided evaluation module for pose error metrics, matching precision, recall, etc.     |



# load_torch_image

# get_global_desc

# get_image_pairs_exhaustive

# get_image_pairs_shortlist

# detect_aliked

# match_with_lightglue
# import_into_colmap

# prediction

# Training
# Submission
