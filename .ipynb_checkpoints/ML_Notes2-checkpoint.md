# ML Notes 2 — Bias, Variance & Regularization

## 1. Bias and Variance

When building a machine learning model, we want it to learn the actual pattern in the data without memorizing noise.

Two major sources of prediction error are:

- Bias
- Variance

### Bias

Bias is the error caused by making overly simple assumptions about the data. A high-bias model fails to capture the underlying relationship.

**Characteristics:**

- Model is too simple
- Underfits the data
- Performs poorly on both training and testing data

### Variance

Variance is the amount by which the model changes when trained on different datasets. A high-variance model learns not only the pattern but also the noise.

**Characteristics:**

- Model is too complex
- Overfits the data
- Excellent training accuracy
- Poor testing accuracy

### Bias–Variance Tradeoff

There is always a balance between bias and variance.

As model complexity increases:

- Bias decreases
- Variance increases

As model complexity decreases:

- Bias increases
- Variance decreases

**Goal:** Find the optimal complexity where total prediction error is minimum.

**Total Error:**

> Prediction Error = Bias² + Variance + Irreducible Error

where:

- Bias² → error due to wrong assumptions
- Variance → error due to sensitivity to data
- Irreducible Error → noise that cannot be removed

## 2. What Is Regularization?

Regularization is a technique used to reduce overfitting by adding a penalty to the model's coefficients. Instead of minimizing only the prediction error, we also penalize large coefficients. Regularization helps the model become simpler and generalize better.

**Benefits:**

- Prevents overfitting
- Improves generalization
- Reduces model complexity
- Handles multicollinearity
- Produces more stable predictions

## 3. Ridge Regression

Ridge Regression adds the square of coefficients as a penalty. Large coefficients are reduced. No coefficient becomes exactly zero.

**Advantages**

- Reduces overfitting
- Works well with multicollinearity
- Uses all features
- Stable solution

**Disadvantages**

- Does not perform feature selection
- Less interpretable than LASSO

## 4. LASSO Regression

Least Absolute Shrinkage and Selection Operator. Some coefficients become exactly zero.

**Advantages**

- Reduces overfitting
- Performs feature selection
- Produces simpler models
- Easy to interpret

**Disadvantages**

- Can remove useful correlated features
- May be unstable when features are highly correlated

## 5. Elastic Net Regression

Elastic Net combines both **Ridge** and **LASSO** regularization.

**Why Elastic Net?**

Suppose a dataset has:

- Thousands of features
- Many correlated variables

LASSO may remove useful variables. Ridge keeps all variables. Elastic Net balances both approaches.

**Advantages**

- Performs feature selection
- Handles multicollinearity
- More stable than LASSO
- Suitable for high-dimensional datasets

**Disadvantages**

- Requires tuning two parameters
- Slightly more computationally expensive
