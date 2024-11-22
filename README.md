# Project 2

Select one of the following two options:

## Boosting Trees

Implement the gradient-boosting tree algorithm (with the usual fit-predict interface) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1.

Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

## Model Selection

Implement generic k-fold cross-validation and bootstrapping model selection methods.

In your README, answer the following questions:

* Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
* In what cases might the methods you've written fail or give incorrect or undesirable results?
* What could you implement given more time to mitigate these cases or help users of your methods?
* What parameters have you exposed to your users in order to use your model selectors.

See sections 7.10-7.11 of Elements of Statistical Learning and the lecture notes. Pay particular attention to Section 7.10.2.

As usual, above-and-beyond efforts will be considered for bonus points.




# Gradient Boosting Trees Implementation

This project implements a **Gradient Boosting Tree** algorithm, adhering to Sections 10.9–10.10 of *Elements of Statistical Learning (2nd Edition)*. The model uses decision trees as weak learners and iteratively minimizes a loss function to improve predictive performance. It is designed for regression tasks and provides flexibility with custom loss functions, parallelization, and early stopping.

---

## Features
- **Gradient Boosting for Regression**:
  - Supports loss functions:
    - Mean Squared Error (MSE)
    - Custom loss functions for specialized tasks
- **Parallel Tree Training**:
  - Utilizes `joblib` for efficient parallel execution.
- **Early Stopping**:
  - Stops training if validation loss stops improving, reducing overfitting.
- **Visualization Tools**:
  - Tracks validation loss progression over iterations.
  - Displays feature importance to highlight influential predictors.
- **Benchmarking**:
  - Compares training times and accuracy (MSE) for parallel and non-parallel implementations.

---

# Project Checklist

| Requirement | Fulfilled? |
| ----- | ----- |
| Fit-Predict Interface | ✅ |
| Gradient Boosting Mechanism | ✅ |
| Loss Functions (mse, logloss) | ✅ |
| Parallel Training | ✅ |
| Early Stopping | ✅ |
| Feature Importance Visualization | ✅ |
| Validation Loss Visualization | ✅ |
| Comparison: Parallel vs Non-Parallel | ✅ |
| Support for Sparse Data | ✅ |
| Custom Loss Functions | ✅ |

---

## How to Run the Code

1. **Set Up the Environment**:
   - Install dependencies with the following command:
     ```bash
     pip install numpy matplotlib scikit-learn joblib scipy
     ```
   - Ensure you are using **Python 3.8+** and running the code in **Jupyter Notebook**.

2. **Execute the Code**:
   - Copy and paste the script into a Jupyter Notebook cell.
   - Run the `if __name__ == "__main__":` section to:
     - Train the model
     - Benchmark parallel vs non-parallel versions
     - Generate visualizations for validation loss and feature importance.

3. **Usage Example**:
   Run the benchmarking function for both parallel and non-parallel versions:
   - **Non-Parallel Version**:
     ```python
     non_parallel_time, non_parallel_mse = benchmark_training(parallel=False)
     ```
   - **Parallel Version**:
     ```python
     parallel_time, parallel_mse = benchmark_training(parallel=True)
     ```

   Outputs:
   - Training times for both versions
   - Test Mean Squared Error (MSE)
   - Visualizations of validation loss progression and feature importance.

---

## 1. What does the model do, and when should it be used?

The model implements **gradient boosting**, an iterative optimization technique that uses decision trees as weak learners to reduce prediction errors.

### Specific Details
- **Functionality**:
  - Each tree in the ensemble predicts the residuals (errors) from the previous trees.
  - These residuals are scaled by a learning rate and added to improve overall predictions.
- **MSE Loss**:
  - Minimizes the squared error between actual and predicted values.
- **Custom Loss**:
  - Allows flexible optimization objectives for specific regression problems.

### Use Cases
- **Complex and Nonlinear Data**:
  - Suitable for datasets where relationships between features and target are non-linear.
- **Tabular Data**:
  - Ideal for structured data with moderate size.
- **Interpretability**:
  - Provides insights into feature importance for model explainability.

---

## 2. How did you test your model to determine if it is working reasonably correctly?

The model was tested using a combination of synthetic data, benchmarking, and visualization techniques:

1. **Synthetic Data Testing**:
   - Generated regression datasets using `sklearn.datasets.make_regression`.
   - Verified performance (low MSE) on datasets with known inputs and outputs.

2. **Parallel vs Non-Parallel Benchmarking**:
   - Measured and compared training times for parallel and non-parallel implementations.
   - Ensured consistent MSE results between both approaches.

3. **Validation Loss Tracking**:
   - Recorded validation loss over iterations.
   - Confirmed that loss decreased steadily during training.
   - Early stopping tested to verify it halts training when improvements cease.

4. **Visualizations**:
   - Plotted validation loss to observe learning progression.
   - Displayed feature importance to validate significant predictors.

5. **System Compatibility**:
   - Verified compatibility with Jupyter Notebook and required libraries.
   - Hardcoded paths to avoid system-specific issues.

---

## 3. What parameters have you exposed to users to tune performance?

The following parameters allow users to customize the model:

| **Parameter**              | **Description**                                         | **Default** |
|----------------------------|---------------------------------------------------------|-------------|
| `n_estimators`             | Number of boosting iterations.                         | 100         |
| `learning_rate`            | Step size for updates.                                  | 0.1         |
| `max_depth`                | Maximum depth of decision trees.                       | 3           |
| `loss_function`            | Loss function for optimization (`"mse"` or custom).     | `"mse"`     |
| `early_stopping_rounds`    | Stops training if no improvement in validation loss.    | `None`      |
| `n_jobs`                   | Number of parallel jobs for tree training.             | 1           |

### Usage Example
```python
model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42,
    early_stopping_rounds=10, 
    n_jobs=-1
)
model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
```

---

## 4. Are there specific inputs that your implementation has trouble with?

### Challenges

1. **High Dimensionality**:
   - Performance can degrade with very high-dimensional datasets.
   - **Mitigations**: Use feature selection or add regularization.

2. **Imbalanced Data**:
   - Skewed target distributions may reduce accuracy.
   - **Mitigations**: Preprocess data or use custom loss functions.

3. **Sparse Data**:
   - While supported, datasets with extreme sparsity may require specialized preprocessing.

4. **Overfitting**:
   - On small datasets, overfitting may occur with deep trees or too many iterations.
   - **Mitigations**: Limit `max_depth` or use early stopping.

### Workarounds
- Incorporate **cross-validation** to tune hyperparameters.
- Add regularization techniques like limiting `max_depth` or increasing `min_samples_split`.
- Normalize or preprocess imbalanced data.

---

## Sample Outputs

### Training Time Comparison

| **Version**      | **Training Time** | **MSE**    |
|------------------|-------------------|------------|
| **Non-Parallel** | 2.29 seconds      | 1246.62    |
| **Parallel**     | 1.87 seconds      | 1246.62    |

---

### Visualizations

1. **Validation Loss**:
   - Displays how the model learns iteratively by reducing validation errors.
     ![Validation Loss progress](https://github.com/user-attachments/assets/d9d10697-814f-4723-8477-29727b85534c)

2. **Feature Importance**:
   - Highlights the features that contribute most to predictions.
    ![Feature importance](https://github.com/user-attachments/assets/9df613d5-f3d8-4666-a2c2-42bef0672c55)
    ![Time Comparison](https://github.com/user-attachments/assets/ba15eee3-2b9f-4f23-900f-e52b2dddb4e9)


