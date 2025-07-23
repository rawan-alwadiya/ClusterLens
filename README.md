# ClusterLens: Customer Segmentation with DBSCAN, K-Means & Hierarchical Clustering

## Project Overview
**ClusterLens** explores unsupervised learning algorithms for segmenting wholesale customers using a combination of density-based, partitioning, and hierarchical clustering approaches. The goal is to compare how different clustering algorithms perform in identifying meaningful customer segments based on purchasing behavior.

The workflow starts with **Exploratory Data Analysis (EDA)** and **outlier detection** using **DBSCAN**, which identifies anomalies based on local density. Outliers are then removed to analyze their impact on clustering performance.

Next, **K-Means** and **Agglomerative (Hierarchical) Clustering** (with complete, average, and ward linkage methods) are applied both **with and without outliers** to compare segmentation quality.

---

## Project Objectives
- Apply **DBSCAN** for multivariate outlier detection and determine optimal epsilon using **k-distance elbow method**
- Analyze the impact of outlier removal on clustering results
- Apply **K-Means Clustering**, with **Elbow Method** to determine the optimal number of clusters
- Apply **Agglomerative Clustering** using multiple linkage strategies
- Use evaluation metrics to measure clustering quality
- Visualize cluster separation and interpret segment-level spending patterns

---

## Clustering Evaluation Metrics
- **Silhouette Score** – measures cohesion and separation (range: -1 to 1, higher is better)
- **Davies–Bouldin Index** – evaluates intra-cluster similarity (lower is better)
- **Calinski–Harabasz Score** – reflects cluster compactness and separation (higher is better)

These metrics provide complementary insights into cluster structure, shape, and separation.

---

## Dataset
- **Source**: [Wholesale Customers Data Set – UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wholesale+customers)
- **Samples**: 440
- **Features**:
  - Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen (annual spending)
  - Categorical indicators: Channel (Horeca/Retail), Region (Lisbon, Oporto, Other)

---

## Technologies Used
**Language**: Python  
**Libraries & Tools**:
- `scikit-learn` – clustering algorithms & evaluation metrics
- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – data visualization

**Clustering Techniques**:
- DBSCAN (Density-Based Spatial Clustering)
- K-Means (Partitioning Method)
- Agglomerative Clustering (Hierarchical – Ward, Complete, Average linkage)

---

## Project Impact
This project provides a structured guide for:
- Marketing teams and data analysts exploring customer profiling techniques
- Evaluating the effect of **outliers** on unsupervised learning results
- Comparing **multiple clustering algorithms** under different data conditions
- Generating **interpretable visualizations** and quantitative metrics for data-driven segmentation

**Use Case**: Supports customer targeting strategies, loyalty program design, and personalized marketing campaigns by identifying key customer segments (e.g., high-value buyers, niche spenders, etc.)

---

## Project Links
- GitHub: [github.com/rawan-alwadiya/ClusterLens](https://github.com/rawan-alwadiya/ClusterLens)
- Kaggle Notebook: [https://www.kaggle.com/code/rawanalwadeya/clusterlens-clustering-for-customer-segmentation?scriptVersionId=252068149]
