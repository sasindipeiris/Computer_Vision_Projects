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

Purpose: Loads an image from a file path and converts it into a PyTorch tensor using Kornia for standardized processing.

Detailed Steps:

* Uses K.io.load_image from Kornia with ImageLoadType.RGB32 to load the image as a float32 RGB tensor.

* Adds a batch dimension to make the shape (1, C, H, W).

* Returns the tensor on the specified device.

Returns: A single image as a batch tensor, useful for feeding directly into PyTorch or Kornia models.

# get_global_desc

Purpose:Computes global image descriptors for all input images using a pretrained DINOv2 transformer model.

Detailed Steps:

* Loads the Hugging Face DINOv2 processor and model.

* Iterates over each filename:

* Extracts filename and loads image as a tensor using load_torch_image().

* Prepares input for the transformer model.

* Performs forward pass to get embeddings.

* Applies max pooling and L2 normalization to extract global descriptor.

* Concatenates all descriptors to create a (N, D) tensor.

# get_image_pairs_exhaustive

Purpose:Creates a list of all unique image index pairs (i, j) for exhaustive pairwise comparisons.

Detailed Steps:

* Iterates over each image index i.

* For every i, pairs it with all j > i.

* Skips duplicates and self-pairs.

Returns:A list of all unique index pairs for exhaustive comparisons.

# get_image_pairs_shortlist

Purpose:Generates a list of image pairs likely to match by filtering based on DINOv2 descriptor similarity.

Detailed Steps:

* If dataset has few images, returns all pairs using get_img_pairs_exhaustive().Otherwise:

* Computes descriptors using get_global_desc().

* Computes Euclidean distances between all image descriptors.

* Flags pairs under similarity threshold sim_th.

* Ensures each image has at least min_pairs close neighbors.

* Removes duplicate and overly distant pairs.

Returns: List of image index pairs suitable for further feature matching.

# detect_aliked

Purpose:Detects keypoints and extracts local features using the ALIKED feature extractor, storing them in HDF5 format.

Detailed Steps:

* Initializes the ALIKED model.

* Creates the feature directory if missing.

* Opens two HDF5 files to store keypoints and descriptors.

* Iterates over each image path:

* Loads the image as a tensor.

* Runs ALIKED to get keypoints and descriptors.

* Reshapes and saves results in the HDF5 files.

Returns:None (but stores results on disk).

# match_with_lightglue

Purpose:This function performs feature matching between pairs of images using the LightGlue matcher configured for ALIKED features. It reads keypoints and descriptors stored in HDF5 format, computes matches for each image pair, and stores valid match results in a new HDF5 file.

Detailed Steps:
 
* Initialize LightGlue Matcher**: Set parameters to disable confidence filtering and enable multiprocessing if CUDA is available.

* Open HDF5 Files: Read keypoints and descriptors from disk and prepare a file for storing matches.

* Iterate Over Image Pairs:
  
    - Retrieve filenames and corresponding keys for the images.
      
    - Load keypoints and descriptors, moving them to the computation device.
     
    - Run LightGlue inference to compute distances and matching indices.

* Filter and Store Matches:
  
    - Skip image pairs with no matches.
      
    - Print the number of matches if verbosity is enabled.
      
    - Store matches in the output HDF5 file only if they exceed the `min_matches` threshold.
 
Returns:None. The function writes the match results directly into the `matches.h5` file in the feature directory.

# import_into_colmap

Purpose:This function imports feature data (keypoints and matches) into a COLMAP-compatible SQLite database. It initializes the database structure and populates it with image and match information necessary for COLMAP's 3D reconstruction pipeline.

Detailes Steps:1. 

* Database Connection:
   - Connects to the COLMAP database located at `database_path`.
     
   - Creates the necessary tables (images, keypoints, descriptors, matches, etc.) if they   do not exist.

* Keypoint Importing:
   - Uses the `add_keypoints` utility to read keypoints from HDF5 and associate them with each image.
     
   - Also defines the camera model (`simple-pinhole`) and whether to use a single shared camera model.

* Match Importing:
   - Uses the `add_matches` utility to read match data from HDF5 and add corresponding image-pair relationships to the database.

* Finalization:
  
   - Commits all the changes to the database to ensure data is saved.

Returns: None. All changes are committed to the COLMAP SQLite database at `database_path`.

# prediction

Purpose:Defines a data structure to hold per-image metadata and manages the loading of test or training sample submission entries for the Image Matching Challenge 2025.

# Key Specialities

* **Hybrid Matching Strategy**

  * Combines *global descriptors* (DINOv2) for shortlist selection with *local keypoint detection* (ALIKED) and *learned matching* (LightGlue).

* **Efficient Shortlisting**

  * Uses DINOv2 to compute global feature embeddings and shortlist the most similar image pairs, reducing computational load.

* **Learned Local Feature Detection**

  * Uses ALIKED to extract robust, lightweight keypoints and descriptors, optimized for visual localization and matching.

* **Geometrically-Aware Matching**

  * LightGlue ensures high-quality matching with attention-based filtering and geometric consistency checks.

* **Structured Data Management**

  * Saves keypoints, descriptors, and matches in `.h5` files; integrates with COLMAP-compatible databases using `h5_to_db`.

* **COLMAP Integration**

  * Builds full 3D reconstructions using PyColmap with incremental mapping and RANSAC-based geometric validation.

* **Performance Tracking**

  * Includes detailed timing breakdowns (shortlisting, feature detection, matching, RANSAC, reconstruction) for profiling.

* **Submission Ready**

  * Automatically formats prediction outputs into the required CSV format for Kaggle submission (both train and test logic).

* **Modular + Scalable**

  * Each function is cleanly separated for feature extraction, matching, database import, and reconstruction — easy to modify, debug, or parallelize.






