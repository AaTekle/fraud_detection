# Credit Card Fraud (Anamoly) Detection using Supervised Machine Learning

## Overview

This project builds a fraud detection pipeline using anomaly detection and supervised machine learning techniques on highly imbalanced financial transaction data.

**Goal:** Identify fraudulent credit card transactions while minimizing false positives and maximizing recall for fraud cases.

The project evaluates four machine learning algorithms:

1. Isolation Forest
2. One-Class SVM
3. Local Outlier Factor (LOF)
4. Random Forest (Supervised Baseline)

---

# Problem

Credit card fraud causes billions of dollars in losses annually. Traditional rule-based systems struggle to detect evolving fraud patterns and unknown attack strategies.

The objectives of this project are:

* Detect anomalous transactions in near real-time
* Reduce financial losses from fraud
* Minimize false positives to avoid blocking legitimate customers
* Evaluate unsupervised anomaly detection methods for rare-event detection
* Compare unsupervised learning against a supervised baseline

---

# Dataset

Dataset: Credit Card Fraud Detection Dataset by ULB Machine Learning Group

Source:

* Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dataset Characteristics

| Metric                  | Value   |
| ----------------------- | ------- |
| Total Transactions      | 284,807 |
| Fraudulent Transactions | 492     |
| Fraud Rate              | 0.172%  |
| Features                | 30      |
| Target Column           | Class   |

## Features

The dataset contains:

* `Time`
* `Amount`
* `V1` to `V28`
* `Class`

Features `V1-V28` are anonymized columns generated using PCA for confidentiality.

Target labels:

* `0` = Legitimate transaction
* `1` = Fraudulent transaction
---

# Supervised Learning Algorithms

## 1. Isolation Forest

Isolation Forest isolates anomalies by randomly partitioning data.

Fraudulent points are easier to isolate because they are rare and significantly different from normal observations.

## Mathematical Function

The anomaly score is:

```math
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
```

Where:

| Symbol                      | Meaning                                  | Intuition                                                  |
| --------------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| $s(x,n)$                    | Anomaly score for point (x)              | Final score determining how anomalous a transaction is     |
| $x$                         | Data point                               | A single transaction sample                                |
| $(n)$                         | Number of samples                        | Total observations in the dataset                          |
| $h(x)$                      | Path length of point (x)                 | Number of splits required to isolate the point             |
| $E(h(x))$                   | Expected path length of point (x)        | Average isolation depth across all random trees            |
| $c(n)$                      | Average path length normalization factor | Normalizes path lengths based on dataset size              |
| $-\frac{E(h(x))}{c(n)}$     | Normalized isolation depth               | Measures how quickly the point becomes isolated            |
| $2^{-\frac{E(h(x))}{c(n)}}$ | Exponential anomaly scaling              | Converts path length into a probability-like anomaly score |


### Takeaways:

* Shorter path lengths indicate anomalies
* Normal points require more partitions to isolate
* Fraudulent points isolate quickly

---

# 2. One-Class SVM

One-Class SVM learns the boundary of normal transactions and flags anything outside that boundary as anomalous.

## Mathematical Function

Optimization objective:

```math
\min_{w,\rho,\xi} \frac{1}{2}\|w\|^2 + \frac{1}{\nu n}\sum_{i=1}^{n}\xi_i - \rho
```

Subject to:

```math
(w \cdot \phi(x_i)) \geq \rho - \xi_i
```

| Symbol                               | Meaning                                        | Intuition                                                                 |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------- |
| $x_i$                                | The (i)-th transaction sample                  | A single transaction represented as a feature vector                      |
| $n$                                  | Number of training samples                     | Total normal transactions used during training                            |
| $w$                                  | Hyperplane normal vector                       | Defines the orientation of the decision boundary                          |
| $w^2$                           | Squared Euclidean norm of (w)                  | Controls model complexity and margin smoothness                           |
| $\rho$                               | Decision boundary offset                       | Determines the boundary position relative to the origin                   |
| $\xi_i$                              | Slack variable for sample (i)                  | Measures how much a sample violates the boundary                          |
| $\nu$                                | Fraction of expected anomalies                 | Controls the upper bound of anomalies and lower bound of support vectors  |
| $\phi(x_i)$                          | Kernel transformation function                 | Maps data into a higher-dimensional space                                 |
| $w \cdot \phi(x_i)$                  | Dot product between (w) and transformed sample | Measures the position of a transaction relative to the decision boundary  |
| $\sum_{i=1}^{n}\xi_i$                | Total slack penalty                            | Penalizes boundary violations across all samples                          |
| $\frac{1}{\nu n}\sum_{i=1}^{n}\xi_i$ | Weighted slack penalty term                    | Controls how strongly violations are penalized relative to dataset size   |
| $\frac{1}{2}w^2$ | Regularization term                            | Encourages simpler and smoother decision boundaries to reduce overfitting |

## Interpretation of the Objective Function

The optimization objective consists of three competing components:

1. **Regularization Term**

```math
\frac{1}{2}\|w\|^2
```

Keeps the decision boundary smooth and prevents overfitting.

2. **Slack Penalty**

```math
\frac{1}{\nu n}\sum_{i=1}^{n}\xi_i
```

Penalizes transactions that fall outside the learned normal region.

3. **Boundary Expansion Term**

```math
-\rho
```

Encourages the model to place the boundary as far away from the origin as possible while still enclosing the majority of normal observations.

The constraint:

```math
(w \cdot \phi(x_i)) \geq \rho - \xi_i
```

Ensures that most training samples remain inside the learned normal region. Samples violating this constraint incur a slack penalty through $\xi_i$.

## Application in This Project

The model is trained only on legitimate transactions:

```python
normal_train = X_train[y_train == 0]
```

using:

```python
OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.00172
)
```

where:

```math
\nu = 0.00172
```

corresponds approximately to the fraud rate in the dataset:

```math
\frac{492}{284807} \approx 0.001727
```

This value tells the model that only a very small fraction of observations are expected to behave like anomalies, producing a tight boundary around normal transaction behavior.


---

# 3. Local Outlier Factor (LOF)

Local Outlier Factor (LOF) measures how isolated a transaction is relative to the density of its surrounding neighborhood.

Unlike global anomaly detection methods, LOF compares a transaction only with nearby transactions. This makes it effective for detecting fraud that appears unusual within a local region of the feature space, even if it would not appear anomalous globally.

Fraudulent transactions often occur in sparse local regions where neighboring observations are much less dense than those surrounding legitimate transactions.

## Mathematical Function

The Local Outlier Factor of observation (A) is:

```math
LOF_k(A)=\frac{\sum_{B\in N_k(A)}\frac{lrd(B)}{lrd(A)}}{|N_k(A)|}
```

An equivalent form is:

```math
LOF_k(A)=\frac{1}{|N_k(A)|}\sum_{B\in N_k(A)}\frac{lrd(B)}{lrd(A)}
```

which represents the average ratio between the local density of neighboring points and the local density of point (A).

---

| Symbol | Meaning | Intuition |
|---|---|---|
| $A$ | Observation being evaluated | The transaction whose anomaly score is being calculated |
| $B$ | Neighboring observation | One nearby transaction surrounding $A$ |
| $k$ | Number of nearest neighbors | Controls the size of the local neighborhood used for comparison |
| $N_k(A)$ | Set of $k$-nearest neighbors of $A$ | Transactions closest to $A$ in feature space |
| $B \in N_k(A)$ | $B$ belongs to the neighbor set of $A$ | The summation loops through each neighbor $B$ around $A$ |
| $N_k(A)$ | Number of neighbors in $N_k(A)$ | Usually equal to $k$ |
| $lrd(A)$ | Local reachability density of $A$ | Measures how dense the region around $A$ is |
| $lrd(B)$ | Local reachability density of neighbor $B$ | Measures how dense the region around neighbor $B$ is |
| $\frac{lrd(B)}{lrd(A)}$ | Density ratio | Compares neighbor density to the density around $A$ |
| $\sum_{B\in N_k(A)}$ | Sum over all neighbors of $A$ | Adds one density ratio for each neighbor $B$ |
| $\sum_{B\in N_k(A)}\frac{lrd(B)}{lrd(A)}$ | Total density-ratio sum | Adds all neighbor-to-$A$ density comparisons |
| $\frac{1}{N_k(A)}$ | Mean-scaling factor | Divides the total by the number of neighbors |
| $\frac{\sum_{B\in N_k(A)}\frac{lrd(B)}{lrd(A)}}{N_k(A)}$ | Average density ratio | The mean of all neighbor-to-$A$ density ratios |
| $LOF_k(A)$ | Local Outlier Factor score | Final anomaly score for transaction $A$ |

---

## Step 1: Find the k-Nearest Neighbors

For every transaction (A), LOF first identifies its (k) nearest neighboring transactions:

```math
N_k(A)
```

using a distance metric such as Euclidean distance.

For example:

```text
Transaction A
 ├── Neighbor 1
 ├── Neighbor 2
 ├── Neighbor 3
 ├── ...
 └── Neighbor k
```

These neighbors form the local region used for density estimation.

The parameter used are:

```python
LocalOutlierFactor(
    n_neighbors=35,
    contamination=0.00172,
    novelty=True
)
```

meaning each transaction is compared against its 35 closest neighbors.

---

## Step 2: Compute Reachability Distance

LOF does not use ordinary distance directly.

Instead it uses the **reachability distance**:

```math
reach\_dist_k(A,B)
=
\max
\left(
k\text{-distance}(B),
d(A,B)
\right)
```

where:

| Symbol | Meaning | Intuition |
|---|---|---|
| $A$ | Observation being evaluated | The transaction whose neighborhood density is being measured |
| $B$ | Neighboring observation | A nearby transaction used to estimate local density |
| $k$ | Number of nearest neighbors | Controls the size of the local neighborhood |
| $d(A,B)$ | Actual distance between $A$ and $B$ | Direct distance between two transactions in feature space |
| $k\text{-distance}(B)$ | Distance from $B$ to its $k$-th nearest neighbor | Represents the local neighborhood radius around $B$ |
| $\max(k\text{-distance}(B), d(A,B))$ | Maximum of the two distances | Ensures the reachability distance is never smaller than the neighborhood radius of $B$ |
| $reach\_dist_k(A,B)$ | Reachability distance between $A$ and $B$ | Adjusted distance used in density estimation |
| $k\text{-distance}(B), d(A,B)$ | Inputs to the max function | The two candidate distances being compared |
| $\max(\cdot)$ | Maximum operator | Selects the larger of the two candidate distances |

### Why use reachability distance?

Without this adjustment, extremely close neighbors could create artificially high density estimates.

The reachability distance prevents density from becoming unrealistically large by enforcing a minimum distance threshold.

---

## Step 3: Compute Local Reachability Density

The local reachability density (LRD) of transaction (A) is:

```math
lrd(A)
=
\left(
\frac{
\sum_{B\in N_k(A)}
reach\_dist_k(A,B)
}
{|N_k(A)|}
\right)^{-1}
```

This can also be written as:

```math
lrd(A)
=
\frac{
|N_k(A)|
}{
\sum_{B\in N_k(A)}
reach\_dist_k(A,B)
}
```

---

### Symbol Definitions

| Symbol                          | Meaning                     | Intuition                                   |
| ------------------------------- | --------------------------- | ------------------------------------------- |
| $reach_dist_k(A,B)$             | Reachability distance       | Effective distance from (A) to neighbor (B) |
| $\sum reach_dist_k(A,B)$        | Total reachability distance | Overall spread of the neighborhood          |
| $\frac{\sum_{B \in N_k(A)} reach\_dist_k(A,B)}{N_k(A)}$                                          | Average reachability distance | Average spacing between (A) and its neighbors |
| $lrd(A)$                        | Local reachability density  | Inverse of average spacing                  |

---

### Interpretation

If neighbors are very close:

```math
Average\ Distance \downarrow
```

then:

```math
lrd(A) \uparrow
```

which means the region is dense.

If neighbors are far away:

```math
Average\ Distance \uparrow
```

then:

```math
lrd(A) \downarrow
```

which means the region is sparse.

---

## Step 4: Compute the LOF Score

Once densities have been computed, LOF compares the density of (A) to the density of its neighbors:

```math
LOF_k(A)
=
\frac{
1
}{
|N_k(A)|
}
\sum_{B\in N_k(A)}
\frac{
lrd(B)
}{
lrd(A)
}
```

This is the average density ratio.

---

### Understanding the Density Ratio

For a neighbor (B):

```math
\frac{lrd(B)}{lrd(A)}
```

* Greater than 1 if (B) is denser than (A)
* Equal to 1 if densities are similar
* Less than 1 if (A) is denser than (B)

LOF averages these ratios across all neighbors.

---

## Interpretation of LOF Scores

### LOF ≈ 1

```math
lrd(A) \approx lrd(B)
```

The transaction has similar density to its neighbors.

**Meaning:** Normal transaction.

---

### LOF > 1

```math
lrd(A) < lrd(B)
```

The transaction lies in a region less dense than its neighbors.

**Meaning:** Potential anomaly.

---

### LOF ≫ 1

Example:

```math
LOF(A)=5
```

Neighboring transactions are approximately five times denser than the region containing (A).

**Meaning:** Strong anomaly candidate.

---

### LOF < 1

```math
lrd(A) > lrd(B)
```

The transaction lies in a region denser than its neighbors.

**Meaning:** Very typical observation.

---

## Geometric Intuition

### Normal Transaction

```text
● ● ● ●
 ● A ●
● ● ● ●
```

Transaction (A) sits inside a dense cluster.

Its density is similar to its neighbors:

```math
LOF(A)\approx1
```

---

### Fraudulent Transaction

```text
● ● ● ● ● ●

             A
```

Transaction (A) is isolated from nearby observations.

Its local density is much lower than surrounding regions:

```math
LOF(A)\gg1
```

making it a strong anomaly candidate.

---

## Application in This Project

implementation uses:

```python
LocalOutlierFactor(
    n_neighbors=35,
    contamination=0.00172,
    novelty=True,
    n_jobs=-1
)
```

where:

* **n_neighbors = 35** defines the size of the local neighborhood used to estimate density.
* **contamination = 0.00172** indicates the expected fraud rate (approximately 0.172%).
* **novelty = True** enables scoring of unseen transactions after training.
* **n_jobs = -1** allows parallel computation across CPU cores.

The model is trained only on legitimate transactions:

```python
normal_train = X_train[y_train == 0]
```

and learns the density structure of normal transaction behavior.

Transactions appearing in significantly lower-density regions than their neighbors receive larger LOF scores and are flagged as potential fraud.

### Interpretation

* High LOF score indicates strong outlier behavior
* Dense regions correspond to normal transactions
* Sparse regions correspond to anomalies

---

# 4. Random Forest (Supervised Baseline)

Random Forest is an ensemble of decision trees trained using labeled fraud data.

Unlike anomaly detection methods, it directly learns fraud patterns.

## Mathematical Function

Prediction:
```math
\hat{y} = \text{mode}(T_1(x), T_2(x), ..., T_n(x))
```
Where:

| Symbol    | Meaning                  |
| --------- | ------------------------ |
| $(\hat{y})$ | Final prediction         |
| $(T_i(x))$  | Prediction from tree (i) |
| $(n)$       | Number of trees          |

### Interpretation

* Each tree votes on the transaction class
* Final prediction is majority vote

---

# Evaluation Metrics

# ROC-AUC

Measures ranking quality across thresholds.

```math
ROC = \frac{TP}{TP + FN}
```

```math
FPR = \frac{FP}{FP + TN}
```

Where:

| Symbol | Meaning         |
| ------ | --------------- |
| TP     | True Positives  |
| TN     | True Negatives  |
| FP     | False Positives |
| FN     | False Negatives |

Higher ROC-AUC indicates better separation between fraud and legitimate transactions.

---

# PR-AUC

Precision-Recall AUC is more important for highly imbalanced fraud datasets.

```math
Precision = \frac{TP}{TP + FP}
```

```math
Recall = \frac{TP}{TP + FN}
```

High PR-AUC means the model detects fraud effectively while minimizing false alarms.

---

# Results

| Model                  | ROC-AUC | PR-AUC |
| ---------------------- | ------- | ------ |
| Isolation Forest       | 0.9485  | 0.1198 |
| Local Outlier Factor   | 0.8806  | 0.0174 |
| One-Class SVM          | 0.9418  | 0.2081 |
| Random Forest Baseline | 0.9605  | 0.8452 |

---

# Model Analysis

## Isolation Forest

Strengths:

* Strong anomaly separation
* Handles large datasets efficiently
* Good unsupervised baseline

Weaknesses:

* Lower precision due to many false positives

---

## Local Outlier Factor

Strengths:

* Captures local density anomalies

Weaknesses:

* Weak precision-recall performance
* Sensitive to neighborhood selection
* Struggles with high-dimensional data

---

## One-Class SVM

Strengths:

* Best unsupervised precision-recall performance
* Strong fraud recall

Weaknesses:

* Computationally expensive
* Difficult to scale on very large datasets

---

## Random Forest Baseline

Strengths:

* Highest overall performance
* Strong fraud detection precision and recall

Weaknesses:

* Requires labeled fraud data
* Less effective against unseen fraud patterns

---

# Key Findings

1. Fraud detection is an extreme class imbalance problem.
2. PR-AUC is more informative than accuracy.
3. Isolation Forest and One-Class SVM performed well for unsupervised anomaly detection.
4. Random Forest significantly outperformed anomaly detection methods due to access to labeled fraud examples.
5. One-Class SVM achieved the strongest unsupervised fraud recall.

---

## Dataset & Published Paper References

1. Dataset:
   Andrea Dal Pozzolo et al., Credit Card Fraud Detection Dataset
   [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

2. Isolation Forest:
   Liu, Ting, Zhou. Isolation Forest. IEEE ICDM 2008.

3. Local Outlier Factor:
   Breunig et al. LOF: Identifying Density-Based Local Outliers. SIGMOD 2000.

4. One-Class SVM:
   Schölkopf et al. Estimating the Support of a High-Dimensional Distribution. Neural Computation 2001.
