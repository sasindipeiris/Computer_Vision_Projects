# Overview

This project tackles the classic digit recognition problem using machine learning techniques on the MNIST dataset. It is structured to work in a Kaggle environment, where it processes and models handwritten digit images to classify digits from 0 to 9.

# Dataset

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

# Libraries used

Numerical operations (numpy)

Data loading and processing (pandas)

Visualization (matplotlib)

# Data Preprocessing

| Step                        | Parameters                                                                                | Detailed Purpose                                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1. Convert to NumPy Array   | `data = np.array(data)`                                                                   | Ensures the dataset is a NumPy array for fast numerical operations.                                                    |
| 2. Get Dataset Shape        | `m, n = data.shape`                                                                       | Extracts the number of samples (`m`) and features/columns (`n`).                                                       |
| 3. Shuffle Data             | `np.random.shuffle(data)`                                                                 | Randomizes the row order to remove any inherent bias in the data sequence.                                             |
| 4. Extract Dev Set          | `data_dev = data[0:1000].T`<br>`Y_dev = data_dev[0]`<br>`X_dev = data_dev[1:n]`           | Takes the first 1000 samples as a **development (validation) set**. Transposes the data for easier column-wise access. |
| 5. Normalize Dev Features   | `X_dev = X_dev / 255.`                                                                    | Scales pixel values from `[0, 255]` to `[0, 1]` to improve model convergence.                                          |
| 6. Extract Train Set        | `data_train = data[1000:m].T`<br>`Y_train = data_train[0]`<br>`X_train = data_train[1:n]` | Uses the remaining samples as the **training set** and separates features from labels.                                 |
| 7. Normalize Train Features | `X_train = X_train / 255.`                                                                | Normalizes training data to match the dev set. Prevents skewed training.                                               |
| 8. Get Training Size        | `_, m_train = X_train.shape`                                                              | Stores the number of training examples for downstream use (e.g., batching).                                            |


# Neural Network Functions

| **Function**                  | **Purpose**                   | **Explanation / Reasoning**                                                                                                                                                                             |
| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init_params()`               | Initialize weights and biases | Randomly initializes weights and biases for a 2-layer neural network: `W1`, `b1` for hidden layer; `W2`, `b2` for output layer. Values are shifted by `-0.5` to center around zero, aiding convergence. |
| `ReLU(Z)`                     | Activation function           | Applies **ReLU** (Rectified Linear Unit) to the hidden layer output: returns max(0, z). Helps introduce non-linearity and prevents vanishing gradients.                                                 |
| `softmax(Z)`                  | Output activation             | Converts logits to probabilities by exponentiating and normalizing. Used in the output layer to model multi-class classification.                                                                       |
| `forward_prop(...)`           | Forward pass                  | Performs matrix multiplications and applies activation functions to compute the output of the network (`A2`) from input `X`.                                                                            |
| `ReLU_deriv(Z)`               | ReLU derivative               | Returns gradient for ReLU: 1 where `Z > 0`, else 0. Used in backpropagation to compute how error changes with respect to ReLU input.                                                                    |
| `one_hot(Y)`                  | Label encoder                 | Converts labels (`Y`) to one-hot encoded format. Necessary to compute loss gradient with respect to softmax output.                                                                                     |
| `backward_prop(...)`          | Backpropagation               | Computes gradients (`dW`, `db`) for all layers using chain rule. Uses the loss derivative from softmax and propagates it backward using ReLU derivatives.                                               |
| `update_params(...)`          | Gradient update               | Performs parameter updates using gradients and learning rate `alpha`. Updates `W1`, `b1`, `W2`, `b2` using gradient descent.                                                                            |
| `get_predictions(A2)`         | Class prediction              | Chooses the index (class) with highest softmax probability in output `A2` as predicted class.                                                                                                           |
| `get_accuracy(...)`           | Model evaluation              | Compares predictions with true labels `Y` and returns the accuracy score.                                                                                                                               |
| `gradient_descent(...)`       | Full training loop            | Initializes parameters, trains the network for a fixed number of iterations using forward and backward propagation, updating weights each time. Prints progress every 10 iterations.                    |
| `make_predictions(...)`       | Inference                     | Applies forward propagation using trained weights to generate predictions for given input `X`.                                                                                                          |
| `test_prediction(index, ...)` | Visual check                  | Displays a training image at a specific index and compares the model's prediction with the true label. Helpful for manual verification of model performance.                                            |



# Learning outcomes


**Neural Network Fundamentals**

* Understand how to initialize weights (`W`) and biases (`b`) for a neural network.
* Learn the structure of a simple **2-layer neural network**:

  * Input layer → Hidden layer (ReLU) → Output layer (Softmax).

---

**Data Preprocessing**

* How to normalize pixel data by scaling to the range \[0, 1].
* How to split data into **training** and **development (validation)** sets.
* How to reshape and transpose datasets to fit matrix operations.

---

**Activation Functions**

* Apply and understand **ReLU** activation for hidden layers.
* Use **Softmax** for multi-class probability outputs in the final layer.

---

**Forward and Backward Propagation**

* Implement forward propagation using matrix operations.
* Compute gradients using backpropagation:

  * Chain rule
  * ReLU derivative
  * Softmax + cross-entropy gradient

---

**Training and Optimization**

* Perform **gradient descent** to update weights and biases.
* Understand the role of the **learning rate (`alpha`)** and **iterations**.
* Track performance during training using **accuracy and predictions**.

---

**Model Evaluation and Prediction**

* Use `argmax` to extract predicted classes from softmax outputs.
* Calculate **model accuracy** on the training/dev set.
* Visualize predictions using `matplotlib` to inspect model behavior.

---

**Miscellaneous Skills**

* Implement **one-hot encoding** for class labels.
* Handle NumPy broadcasting and matrix dimensions effectively.
* Build an end-to-end pipeline from data loading to prediction.



