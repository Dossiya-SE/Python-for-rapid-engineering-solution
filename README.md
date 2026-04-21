# Exploratory Data Analysis of Heart Disease Dataset  
**Author:** Dossiya Dakou  
**Course:** EEE 591 – Python for Rapid Engineering Solutions  

---

## Project Motivation  

Understanding relationships between biomedical variables is critical for building reliable predictive models and supporting data-driven decision-making.  

This project applies structured exploratory data analysis (EDA) to a heart dataset to identify statistical dependencies, dominant features, and potential modeling challenges such as multicollinearity.

---

## Objectives  

- Quantify relationships between variables  
- Identify the strongest predictors of the target variable  
- Detect highly correlated features (risk of redundancy)  
- Provide visual and quantitative insights to guide future modeling  

---

## Dataset  

- File: `heart1.csv`  
- Target variable: `a1p2`  
- Contains numerical predictors related to heart-related measurements  

---

## Methodology  

### 1. Data Inspection  
- Dataset structure and types  
- Missing values  
- Summary statistics  

### 2. Correlation Analysis  
- Absolute correlation matrix  
- Extraction of:
  - Top correlated feature pairs  
  - Strongest predictors of the target  

### 3. Covariance Analysis  
- Covariance matrix computation  
- Identification of dominant joint variability patterns  

### 4. Visualization  
- Heatmaps for global structure  
- Pair plots for multivariate relationships  

---

## Key Results  

### Correlation Heatmap  
![Correlation Heatmap](problem1_outputs/corr_heatmap.png)

---

### Covariance Heatmap  
![Covariance Heatmap](problem1_outputs/cov_heatmap.png)

---

### Feature Relationships (Pair Plot)  
![Pairplot](problem1_outputs/pairplot.png)

---

Dossiya Dakou, Master of Science in Engineering in Sustainable Engineering at Arizona State University
