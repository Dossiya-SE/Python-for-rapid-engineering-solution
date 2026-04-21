# %%
# ============================================================
# Name: Dossiya Dakou
# Assignment: EEE 591 – Project 1 Problem 1
# Date: February 18, 2026 (Africa/Kigali)
#
# AI Usage & Citation:
# I used ChatGPT only for conceptual help with mathematics, numerical
# integration, and basic Python syntax. I did NOT use AI to write code for me,
# design algorithms, or debug my code.
# Chat link (public): https://chatgpt.com/share/e/6995c2d2-8258-8004-b971-453abb6da6f2 
# Links I used for help: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html ;
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html ; 
# https://seaborn.pydata.org/generated/seaborn.heatmap.html ; 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
# ============================================================


# %%
import os # for interacting with the operating system, such as file handling and directory management
import numpy as np # for numerical operations and array manipulation
import pandas as pd # for data manipulation and analysis 
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization 
from pandas.plotting import scatter_matrix # for scatter matrix plot


# %%
DATA_PATH = "heart1.csv"   # expects file in same folder as this script
TARGET = "a1p2" # the name of the column in the dataset that contains the target variable (the variable we want to predict or analyze)
OUTPUT_DIR = "problem1_outputs" # the directory where the output files (such as plots and results) will be saved. If this directory does not exist, it will be created by the code.


# %%
os.makedirs(OUTPUT_DIR, exist_ok=True) # This line creates the directory specified by OUTPUT_DIR if it does not already exist. The exist_ok=True argument allows the function to ignore the error that would occur if the directory already exists, ensuring that the code runs smoothly without interruption.

def outpath(fname: str) -> str:
    return os.path.join(OUTPUT_DIR, fname) # This function takes a filename (fname) as input and returns the full path to that file within the OUTPUT_DIR directory. It uses os.path.join to concatenate the OUTPUT_DIR path with the provided filename, ensuring that the resulting path is correctly formatted for the operating system being used.


# %%
# Load the dataset into a pandas DataFrame and print basic information about it, 
# such as its shape (number of rows and columns) and the names of the columns. 
# This helps to understand the structure of the dataset before performing any analysis or visualization.
df = pd.read_csv(DATA_PATH)
print("Dataset shape (rows, cols):", df.shape)
print("Columns:", list(df.columns))


# %%
df.info()

# %%
df.head()

# %%
df.isna().sum()

# %%
df.describe()

# %%
# Print the counts of each unique value in the target column (a1p2) to understand the distribution of the target variable. 
# This is important for identifying class imbalances or understanding the frequency of different outcomes in the dataset.  
print("\nTarget value counts (a1p2):")
print(df[TARGET].value_counts())


# %%
# Calculate the correlation matrix for the numeric columns in the DataFrame and take the absolute values of the correlations.
corr = df.corr(numeric_only=True).abs()


# %%
# Display the first 14 rows of the correlation matrix to examine the relationships between the numeric features in the dataset. 
# This can help identify which features are strongly correlated with each other and with the target variable, 
# which is useful for feature selection and understanding the data.
corr.head(14)


# %%
# Display summary statistics of the correlation matrix, such as mean, standard deviation, minimum, and maximum values.
corr.describe()


# %%
# Save the correlation matrix to a CSV file in the specified output directory. 
# The file will be named "correlation_matrix_abs.csv". 
# This allows for further analysis or sharing of the correlation data outside of this script. 
# The outpath function is used to generate the full path for the output file, ensuring it is saved in the correct location.   
corr.to_csv(outpath("correlation_matrix_abs.csv"))


# %%
# Create a heatmap of the correlation matrix using seaborn. 
# The heatmap will visualize the strength of the correlations between the numeric features in the dataset.
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=False)
plt.title("Correlation Matrix (abs)")
plt.tight_layout()
plt.savefig(outpath("corr_heatmap.png"), dpi=300) # Save the heatmap as a PNG file in the output directory with a resolution of 300 dots per inch (dpi) for high quality.
plt.show() # Display the heatmap on the screen


# %%
# To avoid duplicate pairs in the correlation matrix (since it is symmetric), 
# we can create a new matrix that only contains the upper triangle of the original correlation matrix. 
corr_nodup = corr.copy()
corr_nodup *= np.tri(*corr_nodup.values.shape, k=-1).T # This line creates a copy of the original correlation matrix (corr) and then multiplies it element-wise by a lower triangular matrix (created using np.tri). The np.tri function generates a lower triangular matrix of ones, and the k=-1 argument shifts the diagonal down by one, ensuring that the diagonal and upper triangle of the correlation matrix are set to zero. This effectively removes duplicate pairs of correlations, leaving only the unique pairs in the upper triangle of the matrix.
corr_nodup.head(14) # Display the first 14 rows of the new correlation matrix without duplicate pairs to examine the unique relationships between the numeric features in the dataset.


# %%


corr_unstacked = corr_nodup.unstack()
corr_unstacked = corr_unstacked[corr_unstacked > 0]  # drop zeros
corr_unstacked.sort_values(ascending=False, inplace=True)

# Exclude any pair that includes TARGET so this list is predictor–predictor only
corr_unstacked = corr_unstacked[
    (corr_unstacked.index.get_level_values(0) != TARGET) &
    (corr_unstacked.index.get_level_values(1) != TARGET)
]

top_corr_pairs = corr_unstacked.head(15)
print("\nTop 15 correlated predictor–predictor feature pairs (abs):")
print(top_corr_pairs)


# %%
# --- Correlation with target (use FULL corr so ALL features are included) ---
corr_with_target = corr[TARGET].drop(labels=[TARGET]).sort_values(ascending=False)

print("\nCorrelation with target (abs) - strongest first:")
print(corr_with_target)

corr_with_target.to_csv(outpath("correlation_with_target_abs.csv"), header=["abs_corr"])


# %%

cov = df.cov(numeric_only=True).abs() # Calculate the covariance matrix for the numeric columns in the DataFrame and take the absolute values of the covariances. The covariance matrix shows how much two random variables vary together, and taking the absolute values allows us to focus on the strength of the relationships without considering the direction (positive or negative). This can be useful for understanding the relationships between features in terms of their joint variability.


# %%
cov.head(14) # Display the first 14 rows of the covariance matrix to examine the relationships between the numeric features in terms of their joint variability. This can help identify which features vary together and may have a strong relationship, which can be useful for feature selection and understanding the data.


# %%
cov.describe() # Display summary statistics of the covariance matrix, such as mean, standard deviation, minimum, and maximum values. This helps to understand the overall distribution of the covariances between features in the dataset.


# %%
cov.shape # Display the shape of the covariance matrix to understand its dimensions, which should be the same as the number of numeric features in the dataset. This can help confirm that the covariance matrix has been calculated correctly and includes all relevant features.


# %%
cov.to_csv(outpath("covariance_matrix_abs.csv")) # Save the covariance matrix to a CSV file in the specified output directory.


# %%
# Create a heatmap of the covariance matrix using seaborn.
plt.figure(figsize=(10, 7))
sns.heatmap(cov, annot=False)
plt.title("Covariance Matrix (abs)")
plt.tight_layout()
plt.savefig(outpath("cov_heatmap.png"), dpi=300)
plt.show()


# %%
# Similar to the correlation matrix, we can create a new covariance matrix that only contains the upper triangle of the original covariance matrix to avoid duplicate pairs.
cov_nodup = cov.copy()
cov_nodup *= np.tri(*cov_nodup.values.shape, k=-1).T


# %%


cov_unstacked = cov_nodup.unstack()
cov_unstacked = cov_unstacked[cov_unstacked > 0]  # drop zeros
cov_unstacked.sort_values(ascending=False, inplace=True)

# Exclude any pair that includes TARGET so this list is predictor–predictor only
cov_unstacked = cov_unstacked[
    (cov_unstacked.index.get_level_values(0) != TARGET) &
    (cov_unstacked.index.get_level_values(1) != TARGET)
]

top_cov_pairs = cov_unstacked.head(15)
print("\nTop 15 covarying predictor–predictor feature pairs (abs):")
print(top_cov_pairs)


# %%
# --- Covariance with target (use FULL cov so ALL features are included) ---
cov_with_target = cov[TARGET].drop(labels=[TARGET]).sort_values(ascending=False)

print("\nCovariance with target (abs) - strongest first:")
print(cov_with_target)

# Save the covariance values with the target variable to a CSV file in the specified output directory.
cov_with_target.to_csv(outpath("covariance_with_target_abs.csv"), header=["abs_cov"])


# %%
# Create a pair plot (scatter matrix) of the numeric features in the dataset, colored by the target variable (a1p2). 
# The pair plot will show scatter plots of each pair of features, 
# as well as histograms of each individual feature along the diagonal. 
# The points in the scatter plots will be colored based on the target variable (a1p2) 
# to help visualize how the features relate to the target variable and to each other. 
# This can be useful for identifying patterns, relationships, and potential clusters in the data.   
cvals = df[TARGET].map({1: 0, 2: 1}).to_numpy()
predictors_df = df.drop(columns=[TARGET])

plt.figure()
scatter_matrix(
    predictors_df,
    figsize=(18, 18),
    diagonal="hist",
    alpha=0.6,
    marker=".",
    c=cvals
)
plt.suptitle("Pair Plot (scatter-matrix)", y=0.9)
plt.tight_layout()
plt.savefig(outpath("pairplot.png"), dpi=300)


# %%
# Display the pair plot on the screen. 
# This allows for visual inspection of the relationships between features and the target variable, 
# which can provide insights into the structure of the data and inform further analysis or modeling decisions.    
plt.show() 
print("\nSaved outputs to:", OUTPUT_DIR)
print(" - correlation_matrix_abs.csv")
print(" - correlation_with_target_abs.csv")
print(" - covariance_matrix_abs.csv")
print(" - covariance_with_target_abs.csv")
print(" - corr_heatmap.png")
print(" - cov_heatmap.png")
print(" - pairplot.png")



