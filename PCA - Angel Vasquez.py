# Name: Angel Vasquez
# Lab Partner: Gabirel
import numpy as np                         # Import numerical computing library
import pandas as pd                        # Import pandas to handle tabular data
import matplotlib.pyplot as plt            # Import plotting library for visualization
from sklearn.preprocessing import StandardScaler   # Import scaler to normalize data
from sklearn.decomposition import PCA              # Import PCA for dimensionality reduction

# 1. LOAD DATA
credit_dataset = pd.read_csv('BankChurners.csv')   # Load the credit card customer dataset from CSV file

print("Dataset shape:", credit_dataset.shape)      # Print number of rows and columns in dataset
print(credit_dataset.head())                       # Display first few rows to preview the data

# NEW: SHOW ALL FEATURES
print("\nAll Columns:")
print(credit_dataset.columns.tolist())             # Show every column name in the dataset

numeric_features = credit_dataset.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = credit_dataset.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Features ({}):".format(len(numeric_features)))
print(numeric_features)                            # Show numeric features (PCA-ready after scaling)

print("\nCategorical Features ({}):".format(len(categorical_features)))
print(categorical_features)                        # Show categorical features (need encoding first)

# 2. SEPARATE FEATURES AND LABELS
label_col = 'Attrition_Flag'  # Specify the label column that indicates churn status

# Drop obvious ID column if present
if 'CLIENTNUM' in credit_dataset.columns:
    credit_dataset = credit_dataset.drop(columns=['CLIENTNUM'])   # Remove customer ID since it has no predictive value

print("\nLabel counts:\n", credit_dataset[label_col].value_counts())  # Show how many customers stayed vs churned

features = [col for col in credit_dataset.columns if col != label_col]  # Select all columns except the churn label as features
print("\nFeatures ({}):".format(len(features)), features)               # Print features list like the wine example

# Convert categorical columns to numeric (one-hot encoding)
X = pd.get_dummies(credit_dataset[features], drop_first=True)   # Convert text-based categories into numeric columns

print("\nFeature matrix shape:", X.shape)     # Show updated number of usable numeric features

# 3. NORMALIZE (STANDARDIZE) THE FEATURES
x = StandardScaler().fit_transform(X)    # Scale all features so they have mean=0 and standard deviation=1
print("\nNormalized data — Mean: {:.4f}, Std: {:.4f}".format(np.mean(x), np.std(x)))  # Confirm normalization worked

# 4. APPLY PCA
pca = PCA(n_components=2)                # Create PCA model to reduce data to 2 main components
principal_components = pca.fit_transform(x)   # Apply PCA transformation to normalized data

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])   # Store PCA results in a new dataframe
pca_df[label_col] = credit_dataset[label_col].values   # Add churn label back for visualization

print("\nExplained variance per component:")   # Show how much information each component keeps
print("  PC1: {:.1%}  |  PC2: {:.1%}".format(*pca.explained_variance_ratio_))
print("  Combined: {:.1%} of total variance captured".format(pca.explained_variance_ratio_.sum()))

# 5. VISUALIZE
plt.figure(figsize=(10, 10))        # Create plot canvas
plt.xlabel('Principal Component 1', fontsize=20)   # Label x-axis
plt.ylabel('Principal Component 2', fontsize=20)   # Label y-axis
plt.title('PCA of Credit Card Customers', fontsize=20)   # Add title
plt.xticks(fontsize=12); plt.yticks(fontsize=14)   # Adjust tick font sizes

# Two churn labels → two colors
labels = pca_df[label_col].unique()   # Get unique churn categories
for lab in labels:
    mask = pca_df[label_col] == lab   # Create filter for each category
    plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],  # Plot each group on PCA graph
                s=50, label=str(lab))

plt.legend(prop={'size': 12})   # Display legend to identify customer groups
plt.tight_layout()              # Adjust layout to prevent overlap
plt.savefig('pca_creditcard.png', dpi=150)   # Save PCA plot as image file
plt.show()                      # Display the PCA plot

loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print("\nTop drivers of PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))

print("\nTop drivers of PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head(10))