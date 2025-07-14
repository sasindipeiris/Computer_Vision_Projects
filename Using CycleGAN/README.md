# Overview

This project demonstrates the use of Cycle-Consistent Generative Adversarial Networks (CycleGANs) to translate photos of real-world scenes into the artistic style of Claude Monet’s paintings. CycleGAN is a powerful deep learning architecture that enables unpaired image-to-image translation, meaning it can learn to map images from one domain to another without requiring corresponding image pairs.

# Dataset

The dataset used is the “GAN Getting Started” dataset from Kaggle, which contains two unpaired sets of images:

1. photo_jpg/: Real-world landscape photographs.

2. monet_jpg/: Digitized Monet-style artworks.

These images are used to train the CycleGAN model to translate between the photo and Monet domains.

| **Variable**   | **Path**                                              | **Purpose**                                                                 |
|----------------|--------------------------------------------------------|------------------------------------------------------------------------------|
| `base_dir`     | `/kaggle/`                                             | Root directory for the Kaggle environment.                                  |
| `input_dir`    | `/kaggle/input/gan-getting-started/`                   | Directory containing the input dataset including photo and Monet images.    |
| `output_dir`   | `/kaggle/data/painter/output`                          | Directory to store generated outputs, such as Monet-style images.           |
| `working_dir`  | `/kaggle/data/painter/working`                         | Directory for intermediate files, logs, or checkpoints during training.     |
| `photos_path`  | `/kaggle/input/gan-getting-started/photo_jpg`          | Folder containing the input real-world photographs.                         |
| `monet_path`   | `/kaggle/input/gan-getting-started/monet_jpg`          | Folder containing the target style Monet-style paintings.                   |

# Libaries used

| **Library / Module**                     | **Purpose**                                                                                   |
|------------------------------------------|-----------------------------------------------------------------------------------------------|
| `random`                                 | Generate random numbers, used for shuffling or sampling.                                     |
| `re`                                     | Perform regular expression matching (e.g., for text pattern processing).                     |
| `shutil`                                 | File operations like copying, moving, or deleting files.                                     |
| `zipfile`                                | Extracting or creating `.zip` archives.                                                      |
| `itertools.chain`                        | Flatten or chain multiple iterables together.                                                |
| `os`                                     | Interact with the operating system (path handling, directories, files).                      |
| `time`                                   | Measure code execution time, delays, or timestamps.                                          |
| `typing.Optional`                        | Used for optional type hinting in Python functions.                                          |
| `matplotlib.pyplot`                      | Visualize data using plots or image grids.                                                   |
| `numpy`                                  | Numerical operations and handling arrays/tensors.                                            |
| `tqdm`                                   | Display progress bars during loops or batch processing.                                      |
| `PIL.Image`                              | Load, transform, and save image files.                                                       |
| `torch`                                  | Core PyTorch library for tensor operations and modeling.                                     |
| `torch.nn`                               | Neural network layers like `Conv2d`, `Linear`, etc.                                          |
| `torch.optim`                            | Optimization algorithms like SGD and Adam.                                                   |
| `torch.optim.lr_scheduler`               | Adjust learning rates during training (e.g., step decay).                                    |
| `torch.utils.data.DataLoader`            | Load datasets in batches for efficient training.                                             |
| `torch.utils.data.Dataset`              | Base class for PyTorch datasets.                                                             |
| `torch.utils.data.Subset`                | Create a subset of a dataset (e.g., for validation split).                                   |
| `torchvision.transforms`                 | Preprocessing functions like resizing, normalization.                                        |
| `torchvision.utils`                      | Visual utilities like saving images in a grid.                                               |
| `torch.profiler`                         | Tools for profiling model performance and memory usage.                                      |
| `ProfilerActivity`, `profile`, `record_function` | Specify CPU/GPU activities and profile specific blocks of code.                      |

# EDA

Random samples from both photo_jpg and monet_jpg folders are visualized in a grid.

Dimensions and data types of the images are checked.

This step ensures that data is correctly loaded and visually confirms domain differences (natural photo vs. painted style).

# Image Transformation Pipeline

Image transformations for training and evaluation include:

1. Resizing to 256×256 pixels

2. Random horizontal flipping (for augmentation - only for training images)

3. Conversion to tensors and normalization to [-1, 1] range

Separate pipelines are defined for training and validation/test images to ensure consistent preprocessing

# ImageDataset class

The `ImageDataset` class is a custom PyTorch `Dataset` designed for training CycleGAN models by loading **paired but unaligned images** from two different domains (e.g., photos and Monet paintings). Its main role is to provide batches of image pairs—one from each domain—that can be used to train the generator and discriminator networks.

 

* **Initialization (`__init__`)**:

  * Takes in file paths for images from **Domain X** and **Domain Y** (e.g., real photos and Monet paintings).
  * Lists all the image files in each domain and stores them.
  * Optionally accepts a set of **transformations** (such as resizing or normalization) to apply to both domains.
  * Sets the total dataset length as the **minimum** of the two domains' sizes to ensure one-to-one sampling even if the domains are unbalanced.

* **Image Retrieval (`__getitem__`)**:

  * Retrieves a pair of images, one from each domain, using the given index.
  * Uses the **modulo operator** to safely loop through the datasets even if the sizes are unequal.
  * Loads the images using `PIL.Image`, converts them to RGB, and applies any provided transformations.
  * Returns a tuple `(x_image, y_image)` as a **training sample**.

* **Dataset Length (`__len__`)**:

  * Returns the number of paired samples, which is the minimum of the number of images in both domains.

This class ensures a **consistent and flexible way** to load and transform unpaired images for domain translation tasks. It abstracts away the handling of differing dataset sizes and provides ready-to-use image pairs for training CycleGAN models.

# get_loaders function

The get_loaders function is a data loading utility that prepares and returns PyTorch DataLoaders for training and validation in the CycleGAN training process. It handles loading, transforming, batching, and splitting of images from two unaligned domains (e.g., Monet paintings and real photographs), using a reproducible and efficient approach.

**Inputs:**

x_path and y_path: File paths to images from Domain X (e.g., Monet) and Domain Y (e.g., photos).

train_transform and val_transform: Transformation pipelines for training and validation sets.

batch_size: Number of image pairs per batch (default from BATCH_SIZE).

train_split: Fraction of dataset to use for training (default 0.8, overridden to 0.9 in this call).

num_workers: Number of parallel threads for data loading (default from NUM_WORKERS).

**Steps Performed:**

1. Initialize the full dataset using the custom ImageDataset class, with both domains loaded but no transformations applied yet.

2. Calculate training and validation sizes using the train_split ratio.

3. Split the dataset deterministically using torch.Generator().manual_seed() to ensure reproducibility.

4. Create Subset datasets for training and validation, each using its own set of transformations (train_transform or val_transform).

5. Wrap each subset in a PyTorch DataLoader:

  The training loader enables shuffling for randomization.
  
  The validation loader does not shuffle to ensure a consistent evaluation order.
  
  Both loaders use pin_memory=True for faster GPU transfer and num_workers to load data in parallel.
  
**Returns:**

A tuple (train_loader, val_loader) to be directly used in the model training loop.

**Purpose**

This function abstracts away the complexity of loading and preparing unpaired image datasets for GAN training. It ensures:

1. Efficient batch loading.

2. Proper preprocessing.

3. Clean separation of training and validation data.

4. Consistency through fixed random seed splitting.

This structure is critical in deep learning workflows to avoid data leakage, maintain training consistency, and scale across CPUs and GPUs.

# ConvBlock class

**Purpose** :ConvBlock is used to either downsample or upsample image features. This is crucial for building generator and discriminator networks in GANs.

**Downsampling vs Upsampling**

1. If is_downsampling=True (default), the block uses a standard Conv2d layer to reduce spatial dimensions (i.e., height and width).

2. If is_downsampling=False, it uses a ConvTranspose2d (also known as a deconvolution layer) to increase spatial resolution—important in generators for image synthesis.

 **Padding**: In downsampling mode, it uses "reflect" padding to reduce edge artifacts that can occur during convolution, making the model better suited for image data.

**Normalization**:  Applies InstanceNorm2d, which normalizes each feature map individually across spatial dimensions. This stabilizes training, especially in style transfer and GAN tasks.

 **Activation**

1. If use_activation=True (default), applies a ReLU activation to introduce non-linearity.

2. If use_activation=False, uses nn.Identity() which performs no operation (i.e., passes the result as-is).

**Integration**: All layers (Conv2d or ConvTranspose2d, InstanceNorm2d, and optional activation) are added to a list and then combined into a nn.Sequential module. This allows the block to be called like a single layer in a neural network model.

**Forward Method**: Defines how data flows through the block: input x is passed through the sequential layers and the transformed result is returned.

# ResnetBlock class

**Purpose:** The ResnetBlock is designed to maintain the input and output dimensions while improving learning efficiency by using a residual connection—a technique that helps the model avoid vanishing gradients and encourages the network to focus on learning residual changes instead of complete transformations.

**Structure**

The block contains two convolutional layers, implemented using the previously defined ConvBlock class:

Both layers have the same number of input and output channels.

They use a kernel size of 3 and padding of 1 to preserve the spatial resolution of the input.

The second ConvBlock has use_activation=False to avoid unnecessary non-linearity at the end of the residual path, ensuring the residual is linearly added to the input.

These layers are wrapped inside an nn.Sequential container for modular and clean execution.

**Residual(Skip) Connection**

In the forward() method, the output of the convolutional block is added element-wise to the original input x.

This addition forms a shortcut connection, allowing gradients to flow more directly through the network during backpropagation and promoting the learning of subtle changes.

**Advantages**

Helps deeper networks train more effectively.

Preserves spatial information while allowing the network to refine its features.

Frequently used in the middle section of CycleGAN generators to maintain content while transforming style.

# Generator class

The `Generator` class in this notebook defines the architecture for the generator network used in the CycleGAN model. The generator is responsible for learning how to transform images from one domain (e.g., real photos) into another domain (e.g., Monet paintings) while preserving the underlying structure and content. Its design leverages downsampling, residual blocks, and upsampling to capture both global and fine-grained features necessary for high-quality image translation.

 **Purpose**:
  The `Generator` transforms an input image from one domain to a stylized version in another domain. In CycleGAN, there are two such generators: one that maps Domain A → B and another for Domain B → A. This bidirectional structure supports cyclic consistency during training.

 **Architecture Breakdown**:

  1. **Initial Convolution Layer**:

     * The input image (typically 3-channel RGB) is passed through a convolutional layer that uses a **7×7 kernel** with padding, which helps preserve the spatial dimensions while capturing low-level features.
     * Reflection padding is often used here to minimize border artifacts.

  2. **Downsampling Layers**:

     * Two successive `ConvBlock` layers with stride 2 are used to **reduce the spatial resolution** while increasing the number of feature maps.
     * This compression helps the model extract abstract, higher-level representations of the input image.

  3. **Residual Blocks**:

     * A series of `ResnetBlock` instances (typically 6 or 9) operate at the lowest resolution to refine features without changing spatial dimensions.
     * These residual blocks allow the model to preserve information while learning transformations, making them essential for stable and expressive image generation.

  4. **Upsampling Layers**:

     * Mirror the downsampling process using `ConvBlock` layers with transposed convolutions (`ConvTranspose2d`) to **increase the resolution** back to the original size.
     * This step reconstructs the transformed image while maintaining the learned style.

  5. **Final Convolution Layer**:

     * Outputs a 3-channel image (RGB) using a **tanh activation function** to ensure the pixel values are scaled between \[-1, 1], which is standard practice for GANs to stabilize training.

  6. **Forward Pass**:

  * The image is passed sequentially through all layers: initial conv → downsampling → residuals → upsampling → final conv.
  * The resulting image is the stylized output suitable for domain translation.

The generator's architecture is tailored for **image-to-image translation tasks**. It balances global structure understanding (via downsampling and residuals) with high-resolution synthesis (via upsampling). The use of residual blocks improves learning stability and quality, while the modular design (built from `ConvBlock` and `ResnetBlock`) makes it flexible and easy to modify. This architecture is core to CycleGAN's ability to learn style transfer without requiring paired training data.

# DiscConvBlock class

**Purpose**

The block is used in the discriminator to progressively reduce the spatial dimensions of the image while increasing the feature representation, helping the model distinguish between real and generated image patches.

**Structure**

1. Conv2D Layer: A 2D convolutional layer with a kernel size of 4 and stride defined by the user, with padding set to 1 and reflect padding mode. This setup maintains local detail and minimizes boundary artifacts.

2. Instance Normalization (Optional): When enabled, InstanceNorm2d normalizes feature maps channel-wise, stabilizing training and improving convergence, especially useful in style transfer tasks like CycleGAN.

3. LeakyReLU Activation: A LeakyReLU with a negative slope of 0.2 is used to allow a small gradient when the unit is not active, preventing dying neurons and improving gradient flow.

**Forward Pass**

The input tensor is passed through the block sequentially, with the convolution capturing local features, optional normalization stabilizing them, and the activation introducing non-linearity.

The DiscConvBlock is crucial for building an effective PatchGAN discriminator. By applying these blocks in succession, the discriminator learns to analyze overlapping image patches rather than entire images, encouraging the generator to focus on producing realistic local textures and details. This design contributes significantly to the overall realism and quality of CycleGAN outputs.

# Discrimator class

**Purpose**

The Discriminator class defines a PatchGAN-based discriminator used in the CycleGAN architecture. Its primary role is to distinguish real images from fake ones by analyzing small overlapping patches rather than the entire image. This approach helps focus on texture and fine-grained details, which is especially useful in tasks like style transfer where local consistency matters.

**PatchGAN Architecture**:The discriminator is built to classify whether each local image patch is real or generated, using a series of convolutional layers that reduce spatial dimensions while increasing feature complexity. This patch-based approach encourages the generator to produce realistic textures at the local level.

**Noise Injection**:During training, random noise is added to the input image (self.noise_std = 0.05). This acts as a regularization technique to prevent the discriminator from overfitting to training data and improves generalization by making the model more robust to slight variations.

**Initial Convolution Block**:The model starts with a DiscConvBlock with stride 2 and no instance normalization. This layer extracts initial features while reducing spatial size.

**Intermediate blocks(with Downsampling)**:

A loop creates a sequence of DiscConvBlocks using nn.ModuleList. Each block:

Doubles the number of feature maps (e.g., from 64 → 128 → 256…),

Uses instance normalization (except the first),

Applies a stride of 2 (except the last block, which uses stride 1 to preserve detail).

**Final Convolution**:A single 1×1 convolution (self.final) with output channel = 1 is applied to collapse the feature map to a patch-wise score. This essentially gives a matrix where each element represents the model's confidence that a corresponding patch is real.

**Sigmoid Activation**:The final output is passed through a sigmoid function to squash values into [0, 1], representing the probability of each patch being real.

This discriminator does not classify the whole image as real or fake. Instead, it operates at a patch level, making it more sensitive to local realism — a key characteristic for style transfer tasks like photo-to-Monet generation. The modularity and progressive deepening also help capture complex features while maintaining manageable model size. Regularization with noise ensures better generalization and more stable GAN training.

# ImageBuffer class

The ImageBuffer class is designed to stabilize training in adversarial networks (like GANs) by maintaining a history of previously generated images. Instead of using only the most recent fake images to train the discriminator, it randomly samples from this buffer of past generated images. This prevents the discriminator from overfitting to the latest generator outputs, thus improving convergence and generalization.

**Initialization:** Takes a capacity argument which defines the maximum number of images the buffer can hold. An internal list (self.images) is initialized to store these image tensors.

**Usage:** If the buffer has not yet reached its full capacity, the input image is added directly and returned.

Once full:

With 50% probability, a randomly selected image from the buffer is returned (and replaced by the new one).

Otherwise, the new image is returned directly.

**Purpose:** 

1. Encourages the discriminator to see a diverse distribution of fake images across different iterations.

2. Prevents mode collapse and helps the GAN train more steadily.

# CycleGAN class

**Purpose:** The CycleGAN class encapsulates the full logic for setting up and training a CycleGAN model. CycleGAN is a type of generative adversarial network architecture specifically developed for image-to-image translation without requiring paired training examples. It uses two generators and two discriminators to learn mappings between two domains (e.g., photographs and Monet-style paintings).

Here are detailed summaries for the `ImageBuffer` class and the `CycleGAN` class from the notebook:

 **Generators**:

  * `generator_G`: Translates images from domain X to domain Y.
  * `generator_F`: Translates images from domain Y to domain X.
  * Each generator is a ResNet-based architecture built with convolutional, residual, and upsampling layers.

**Discriminators**:

  * `discriminator_X` and `discriminator_Y`: Each is a PatchGAN-style model that evaluates whether an image belongs to the respective domain.
  * They classify image patches rather than entire images for better detail sensitivity.

 **Optimizers**:

  * Separate Adam optimizers are initialized for both generators and discriminators, allowing independent gradient updates.

 **Image Buffers**:

  * Two instances of `ImageBuffer` are used to store previously generated images for each domain. This improves discriminator training stability.

 **Loss Functions**:

  * **Adversarial Loss**: Encourages generators to fool discriminators.
  * **Cycle Consistency Loss**: Ensures that an image translated to the other domain and back returns close to the original (e.g., X → Y → X ≈ X).
  * **Identity Loss** (optional): Penalizes changes to images that already belong to the target domain (e.g., Y → Y).

 **Training Loop**:

  * Each batch goes through forward passes in both domain directions.
  * Losses are calculated, backpropagated, and the weights updated.
  * Regular logging and saving of visual results are included for monitoring.

# Model training

| **Parameter**                       | **Value**            | **Description**                                                                                                                   |
| ----------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `generator_G`                       | `Generator()`        | Translates images from domain X (e.g., photos) to domain Y (e.g., Monet paintings).                                               |
| `generator_F`                       | `Generator()`        | Translates images from domain Y to domain X (inverse direction of `generator_G`).                                                 |
| `discriminator_X`                   | `Discriminator()`    | Discriminator trained to distinguish real vs. fake images in domain X.                                                            |
| `discriminator_Y`                   | `Discriminator()`    | Discriminator trained for domain Y, penalizing unrealistic Monet-style generations.                                               |
| `lambda_cycle`                      | `10`                 | Weight for the **cycle consistency loss**, ensuring X → Y → X and Y → X → Y mappings preserve the original image.                 |
| `lambda_identity`                   | `5`                  | Weight for the **identity loss**, encouraging the generators to produce unchanged outputs when input is already in target domain. |
| `learning_rate`                     | `2e-4`               | Learning rate for both generator and discriminator optimizers (Adam). Balances convergence speed and stability.                   |
| `beta1`                             | `0.5`                | First moment decay rate for Adam optimizer. Helps stabilize updates by controlling momentum in weight changes.                    |
| `beta2`                             | `0.999`              | Second moment decay rate for Adam optimizer. Typically set high to maintain a long memory of past gradients.                      |
| `epochs`                            | `NUM_EPOCHS`         | Total number of passes through the full training dataset. Determines training duration.                                           |
| `train_loader`                      | `DataLoader`         | Loads batches of photo and Monet images with shuffling, transformation, and prefetching during training.                          |
| `val_loader`                        | `DataLoader`         | Loads validation images (without shuffling) to monitor translation quality after each epoch.                                      |
| `ImageBuffer`                       | (used internally)    | Stores recently generated images and randomly mixes them into discriminator training batches to reduce oscillations.              |
| `num_of_blocks`                     | `3`                  | Number of intermediate convolutional blocks in the discriminator to progressively extract features from patches.                  |
| `noise_std`                         | `0.05`               | Standard deviation for random noise added to input images during training to improve generalization and regularization.           |
| `train_transform` / `val_transform` | `Transform pipeline` | Includes resizing, normalization, and augmentations to standardize training and validation data formats.                          |





