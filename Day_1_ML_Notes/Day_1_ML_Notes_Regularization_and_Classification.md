# Machine Learning Notes

*Regularization · Classification & Evaluation Metrics*

---

## Part 1: Regularization in Machine Learning

### 1. What Is Regularization?

**Definition.** Regularization adds a penalty for model complexity to the training goal. Instead of only reducing the error on the training data, the model also pays a price for large coefficients. This keeps it from fitting noise, so it performs better on new data.

A normal model minimizes just the prediction error. Regularization changes the goal to:

> *Cost = prediction error (RSS) + λ × (penalty on coefficient sizes)*

Here **λ (lambda)** is the strength knob: larger λ = simpler model; λ = 0 = no regularization. In short, regularization trades a little more bias for a big drop in variance.

**Example:** A house-price model with 150 features fits the old sales perfectly but swings wildly on new listings. Adding a regularization penalty shrinks the weights on noisy features, so predictions stay stable on houses it has never seen. The same idea powers credit scoring, fraud detection, and recommendation systems.

### 2. Bias and Variance

A model's expected error on new data splits into three parts:

> *Total Error = Bias² + Variance + Irreducible Error*

**Bias** is error from being too simple — the model **underfits** and is wrong even on training data. **Variance** is error from being too sensitive to the training data — the model **overfits** and captures noise. **Irreducible error** is the natural noise no model can remove.

#### The Bias–Variance Tradeoff

As complexity rises, bias falls but variance rises; as it falls, the reverse happens. You cannot minimize both at once. The goal is the sweet spot that minimizes **total** error.

![Bias-variance tradeoff](images/p1_fig1_bias_variance_tradeoff.png)

*Figure 1.1 — Bias² falls and variance rises with complexity; total error is lowest in the middle.*

![Underfitting vs balanced fit vs overfitting](images/p1_fig2_underfit_vs_overfit.png)

*Figure 1.2 — Underfitting (high bias) vs. a balanced fit vs. overfitting (high variance).*

Regularization is a main tool for controlling this tradeoff: turning up λ moves a model away from overfitting toward the optimal point.

### 3. Why Use Regularization?

- **Prevent overfitting** — stops the model memorizing training noise.
- **Reduce variance** — makes predictions stable across datasets.
- **Handle multicollinearity** — stabilizes coefficients when predictors are correlated.
- **Automatic feature selection** — Lasso zeroes out irrelevant features.
- **Work in high dimensions** — makes problems solvable when features outnumber samples.
- **Improve interpretability** — fewer, smaller coefficients are easier to trust.

### 4. Ridge Regression (L2)

Ridge adds a penalty equal to λ times the sum of the **squared** coefficients.

> *Minimize: RSS + λ × Σ βⱼ²*

- Shrinks all coefficients toward zero, but almost never exactly to zero — every feature is kept.
- Great when predictors are correlated; shares weight across the group.

**Example:** Predicting house price from square footage and number of rooms (which move together) — Ridge keeps both, just with smaller, stable weights.

### 5. Lasso Regression (L1)

**Lasso** = Least Absolute Shrinkage and Selection Operator. It adds a penalty equal to λ times the sum of the **absolute** coefficients.

> *Minimize: RSS + λ × Σ |βⱼ|*

- Can drive some coefficients to **exactly zero** — built-in feature selection.
- Produces a sparse, easy-to-read model.

**Example:** From 500 possible signals, Lasso may keep only the 20 that truly predict customer churn and discard the rest.

### 6. Difference Between Ridge and Lasso

The penalty defines a "budget" region the solution must stay in: a **circle** for Ridge and a **diamond** for Lasso. The diamond's sharp corners sit on the axes, so the solution often lands there — setting a coefficient to exactly zero. A smooth circle has no corners, so Ridge rarely hits zero.

![L1 vs L2 penalty geometry](images/p1_fig3_l1_l2_geometry.png)

*Figure 1.3 — The L1 diamond produces exact zeros (sparsity); the L2 circle does not.*

![Ridge vs Lasso coefficient paths](images/p1_fig4_ridge_lasso_paths.png)

*Figure 1.4 — As λ grows, Ridge shrinks coefficients smoothly while Lasso snaps them to exactly zero.*

| **Aspect** | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| **Penalty term** | λ × Σ βⱼ² (squared coefficients) | λ × Σ \|βⱼ\| (absolute coefficients) |
| **Effect on coefficients** | Shrinks all smoothly toward zero | Can force some to exactly zero |
| **Feature selection** | No — keeps every feature | Yes — drops irrelevant features |
| **Resulting model** | Dense (all features kept) | Sparse (only key features) |
| **Multicollinearity** | Handles it well; shares weight across correlated features | Tends to pick one correlated feature and drop the rest |
| **Best when** | Many features are useful; predictors are correlated | Many features are irrelevant and you want a simpler model |

### 7. Elastic Net

**Elastic Net** combines both penalties (L1 + L2), controlled by a mixing ratio α.

> *Minimize: RSS + λ × [ α × Σ |βⱼ| + (1 − α) × Σ βⱼ² ]*

- Does feature selection (from L1) while keeping the stability of Ridge (from L2).
- Best when features come in correlated groups — it keeps or drops them together, instead of picking one at random like Lasso.

![Elastic Net region](images/p1_fig5_elastic_net.png)

*Figure 1.5 — The Elastic Net region is a rounded diamond — a blend of L1 and L2.*

### 8. How λ Connects Back to Bias and Variance

Tuning λ (usually with cross-validation) is how you walk along the bias–variance curve:

| **Value of λ (lambda)** | **Effect on the model** |
|---|---|
| **λ = 0** | No penalty — same as ordinary regression. Low bias, high variance (overfitting risk). |
| **Small λ** | Light penalty — coefficients only slightly shrunk. |
| **Well-tuned λ** | Balanced — near the lowest total error. Best generalization. |
| **Large λ** | Heavy penalty — coefficients near zero. High bias, low variance (underfitting risk). |

**In one line:** regularization adds a complexity penalty (λ) that trades a little bias for a big drop in variance — Ridge shrinks coefficients, Lasso also removes features, and Elastic Net does both.

---

## Part 2: Classification & Evaluation Metrics

### 9. How to Reduce Variance

High variance means the model **overfits**. Ways to reduce it:

- **Use simpler models** — fewer parameters, less room to memorize noise.
- **Train on more data** — more examples make the noise average out.
- **Use regularization** — penalties keep the coefficients small.
- **Use cross-validation** — split the data into parts to test fairly and tune so it doesn't overfit one split.

**Example:** A house-price model is perfect on old sales but wrong on new listings. Adding more data and regularization fixes it.

![Learning curve](images/p2_fig1_learning_curve.png)

*Figure 2.1 — More data shrinks the gap between training and test error, so variance goes down.*

### 10. How to Reduce Bias

High bias means the model **underfits**. Ways to reduce it:

- **Use more complex models** — add features or use a tree / neural network.
- **Create synthetic data (up-sampling / down-sampling)** — when one class is rare, add more of it or reduce the common one so the model learns both fairly.

**Example:** A fraud model marks everything "not fraud" because only 1% is fraud. Up-sampling the fraud cases helps it learn what fraud looks like.

### 11. Classification

Classification means predicting **discrete values** (categories), not numbers.

- **Binary** — two classes (spam / not spam).
- **Multi-class** — many classes (cat, dog, bird).

**Example:** Email becomes spam or not spam. A loan is approved or rejected.

### 12. Logistic Regression

Logistic regression is a **classification** method. It gives a **probability**, then labels the data based on it.

**Example:** Will a customer leave the service? The model outputs a probability like 0.82, so the label is "will leave."

#### 12.1 Sigmoid Function

The sigmoid squashes any number into a value between 0 and 1, so it can be read as a probability. Used for **two classes**.

> *σ(z) = 1 / (1 + e⁻ᶻ)*

**Threshold Value**

We fix a cut-off (usually **0.5**). Above it the label is Class 1, below it Class 0.

**How to choose it?** Look at the **ROC curve** and pick the cut-off that best balances catching positives vs. false alarms.

**Example:** In cancer screening, missing a patient is dangerous, so a lower threshold flags more people for tests.

![Sigmoid curve with threshold](images/p2_fig2_sigmoid.png)

*Figure 2.2 — The sigmoid curve; the threshold line decides the final class.*

#### 12.2 Softmax Function

Softmax is like sigmoid but for **more than two classes**. It gives a probability for each class, all adding up to 1; the highest one wins.

> *softmax(zᵢ) = e^{zᵢ} / Σ e^{zⱼ}*

![Softmax probabilities](images/p2_fig3_softmax.png)

*Figure 2.3 — Softmax turns raw scores into probabilities that sum to 1.0.*

**Example:** Recognizing a digit (0–9): softmax gives 10 probabilities, and the biggest one is the answer.

### 13. Confusion Matrix

A confusion matrix compares **predictions vs. truth**. A good model keeps False Positives and False Negatives low.

- **TP** — said YES, and it is YES.
- **TN** — said NO, and it is NO.
- **FP** — said YES, but it is NO (false alarm).
- **FN** — said NO, but it is YES (a miss).

![Confusion matrix](images/p2_fig4_confusion_matrix.png)

*Figure 2.4 — The four cells of a confusion matrix, with Type 1 and Type 2 errors marked.*

| **Metric** | **Formula** | **Simple question it answers** |
|---|---|---|
| **Accuracy** | (TP+TN)/All | How often is the model correct overall? |
| **Precision** | TP/(TP+FP) | When it says YES, how often is it right? |
| **Recall** | TP/(TP+FN) | Of all real YES cases, how many did it catch? |

**Precision, Recall, and Accuracy (Explained)**

All three are calculated from the confusion matrix. To see how they differ, use one running example: a model screens 100 patients for a disease. 20 actually have it, 80 are healthy. The model produces:

- **TP = 18** — sick patients correctly caught
- **FN = 2** — sick patients missed
- **FP = 10** — healthy patients wrongly flagged
- **TN = 70** — healthy patients correctly cleared

**Accuracy** — of ALL predictions, how many were correct. Formula: (TP + TN) / Total = (18 + 70) / 100 = **88%**.

**Example:** A weather model that is right on 88 of 100 days has 88% accuracy. (But accuracy can mislead when a class is rare — see the note.)

**Precision** — of all cases the model labelled positive, how many were truly positive ("when it says YES, how often is it right?"). Formula: TP / (TP + FP) = 18 / 28 = **64%**.

**Example:** In a spam filter, high precision means an email marked spam almost always is spam, so real emails are rarely lost.

**Recall (Sensitivity)** — of all the cases that are truly positive, how many did the model catch ("of all real YES cases, how many did it find?"). Formula: TP / (TP + FN) = 18 / 20 = **90%**.

**Example:** In cancer detection, high recall is critical — the test catches almost everyone who is sick, even at the cost of a few false alarms.

**Why accuracy alone is not enough:** a fraud model where 1% of cases are fraud can label everything "not fraud" and score 99% accuracy — yet catch zero fraud (0% recall). For rare events, rely on precision and recall.

### 14. Type 1 and Type 2 Errors

**False Positive is a Type 1 error. False Negative is a Type 2 error.**

| **Error** | **Simple meaning** | **Real example** |
|---|---|---|
| **Type 1 = False Positive** | Model says YES, but the truth is NO (false alarm). | A good email is sent to the spam folder. |
| **Type 2 = False Negative** | Model says NO, but the truth is YES (a miss). | A sick patient is told they are healthy. |

Which one is worse depends on the problem: for spam, a False Positive hurts; for disease, a False Negative is dangerous.

### 15. ROC Curve and AUC

ROC stands for **Receiver Operating Characteristic**. It shows how well the model separates the two classes as the threshold changes. **AUC** is the area under the curve — a single score for the whole curve.

- **AUC = 1.0** means a perfect model.
- **AUC = 0.5** means random guessing.
- Higher AUC is better.

![ROC curve](images/p2_fig5_roc_curve.png)

*Figure 2.5 — ROC curve; the closer to the top-left corner, the better.*

**Example:** A credit-default model with AUC 0.88 separates risky and safe customers better than one with AUC 0.79.

### 16. Precision–Recall Curve

This curve shows the trade-off between **Precision** and **Recall** as the threshold changes. It is best for **rare-event** problems.

![Precision-Recall curve](images/p2_fig6_precision_recall_curve.png)

*Figure 2.6 — Precision–Recall curve; useful when the positive class is rare.*

**Example:** Fraud is only about 1% of cases. The PR curve shows how much precision you keep as you try to catch more fraud.

---

## Quick Recap

- **Regularization** — a penalty (λ) that trades bias for less variance; Ridge shrinks, Lasso selects, Elastic Net does both.
- **Reduce variance** — simpler model, more data, regularization, cross-validation.
- **Reduce bias** — complex model, synthetic data (up/down-sampling).
- **Logistic regression** — sigmoid (2 classes), softmax (many); threshold makes the label.
- **Confusion matrix** — TP, TN, FP (Type 1), FN (Type 2) → accuracy, precision, recall.
- **ROC/AUC** — overall ranking power; PR curve is best for rare classes.
