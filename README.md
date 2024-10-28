## Author

- **Name**: Asif Khan
- **Date**: October 2024
- **Module**: Module 11 Challenge

# Unsupervised Machine Learning Challenge: Cryptocurrency Clustering

This repository contains the solution for the **Unsupervised Machine Learning Challenge** using **K-means clustering** and **Principal Component Analysis (PCA)** to classify cryptocurrencies based on their price fluctuations across various timeframes.

## Overview

The goal of this project is to classify cryptocurrencies by analyzing their price changes over intervals of 24 hours, 7 days, 30 days, 60 days, 200 days, and 1 year. By applying unsupervised learning techniques, such as **K-means clustering** and **PCA**, we identify patterns and classify the cryptocurrencies into distinct clusters.

## Objectives

1. **Data Preparation**:
   - Load and inspect cryptocurrency market data.
   - Normalize the data for improved clustering performance.
   
2. **Cluster Analysis**:
   - Apply K-means clustering to the normalized data to classify cryptocurrencies.
   - Determine the optimal number of clusters by evaluating the **Elbow Curve**.

3. **Dimensionality Reduction with PCA**:
   - Reduce the dataset dimensions using **Principal Component Analysis (PCA)** to improve cluster visualization.
   - Re-apply K-means clustering to the PCA-transformed data.
   
4. **Evaluation**:
   - Compare clustering results with and without PCA transformation.
   - Determine the weight of each feature in each principal component.

## Dataset

The data for this project is sourced from `crypto_market_data.csv`. Key columns include:
- **Price Change Percentage Intervals**: 24h, 7d, 30d, 60d, 200d, and 1 year.

## Dependencies

This project requires the following Python libraries:

- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning library for clustering and PCA.
- `matplotlib`: Data visualization.
  
Install the dependencies with:

```bash
pip install pandas scikit-learn matplotlib
```

## Project Structure

|--- Crypto_Clustering.ipynb         # Jupyter Notebook with the complete code for the challenge  
|--- Resources  
│   └-- crypto_market_data.csv       # Dataset used in the project  
|--- README.md                        # Project documentation  

## Code Walkthrough

1. **Data Preprocessing**

   - **Data Loading**: The dataset is loaded, and initial exploration is conducted to understand its structure.
   - **Data Normalization**: The `StandardScaler` module from scikit-learn is used to normalize the dataset.

2. **Finding the Optimal Number of Clusters (k) Using K-means**

   - We test multiple values of `k` (from 1 to 11) and plot the Elbow Curve to identify the best `k` for clustering.

3. **K-means Clustering on Original Data**

   - Based on the Elbow Curve, the optimal `k` value is chosen to fit a K-means model on the normalized data.
   - Clustering results are visualized using a scatter plot with the `price_change_percentage_24h` and `price_change_percentage_7d` features.

4. **Dimensionality Reduction with PCA**

   - **PCA Transformation**: The PCA model is set to 3 components to reduce dimensionality, retaining key information while simplifying the dataset.
   - **Explained Variance**: The explained variance ratio of each principal component is calculated to evaluate how much information each component holds.

5. **Finding Optimal k Using PCA Data**

   - The optimal `k` is again determined using the Elbow Curve, now applied to the PCA-transformed data.
   - A K-means model is fit on the PCA data, and clustering results are visualized.

6. **Feature Influence on Principal Components**

   - The weight (influence) of each feature on each principal component is calculated to understand the contribution of each original feature in the PCA components.
  
## Results

1. **Elbow Curve Analysis**: The Elbow Curve for both the original and PCA-transformed data helps identify the best number of clusters for effective classification.
2. **Clustering Visualization**: Scatter plots show clusters formed using both original and PCA-reduced data.
3. **PCA Components**: The explained variance and feature weights in each principal component give insights into the most influential features.

## Usage

To run the project:

1. Clone this repository.
2. Open the Jupyter Notebook `Crypto_Clustering.ipynb`.
3. Run each cell step-by-step to reproduce the results and visualizations.

## Author's Environment Details

### Environment Details
- **Python Implementation**: CPython
- **Python Version**: 3.10.14
- **IPython Version**: 8.25.0
- **Compiler**: Clang 14.0.6
- **Operating System**: Darwin
- **Release**: 23.4.0
- **Machine**: arm64
- **Processor**: arm
- **CPU Cores**: 8
- **Architecture**: 64-bit

### Installed Packages
- **requests**: 2.32.2
- **watermark**: 2.5.0
- **IPython**: 8.25.0
- **ipywidgets**: 8.1.5
- **numpy**: 1.26.4
- **json**: 2.0.9
- **xarray**: 2023.6.0
- **pandas**: 2.2.2

### System Information
- **sys**: 3.10.14 (main, May 6 2024, 14:42:37) [Clang 14.0.6]

## Conclusion

This project demonstrates the use of Unsupervised Machine Learning techniques, specifically K-means clustering and PCA, to classify cryptocurrencies based on their market behavior. By reducing dimensions, we enhance the interpretability and visualization of clusters, helping reveal significant patterns in cryptocurrency price changes over different timeframes.

## Challenge Instructions

Instructions
1. Rename the Crypto_Clustering_starter_code.ipynb file as Crypto_Clustering.ipynb.
2. Load the crypto_market_data.csv into a DataFrame and set the index to the “coin_id” column.
3. Get the summary statistics to see what the data looks like before proceeding.
Prepare the Data
1. Use the StandardScaler() module from scikit-learn to normalize the data from the CSV file.
2. Create a DataFrame with the scaled data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.
• The first five rows of the scaled DataFrame should appear as follows:

Find the Best Value for k Using the Original Scaled DataFrame
Use the elbow method to find the best value for k by completing the following steps:
1. Create a list with the number of k values from 1 to 11.
2. Create an empty list to store the inertia values.
3. Create a for loop to compute the inertia with each possible value of k.
4. Create a dictionary with the data to plot the elbow curve.
5. Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
6. Answer the following question in your notebook: What is the best value for k?
Cluster Cryptocurrencies with K-Means Using the Original Scaled Data
Use the following steps to cluster the cryptocurrencies for the best value for k on the original scaled data:
1. Initialize the K-means model with the best value for k.
2. Create an instance of K-means, define the number of clusters based on the best value of k, and then fit the model using the original scaled DataFrame.
3. Predict the clusters to group the cryptocurrencies using the original scaled DataFrame.
4. Create a copy of the original data and add a new column with the predicted clusters.
5. Create a scatterplot using pandas’ plot as follows:
• Set the x-axis as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d".
Optimize Clusters with Principal Component Analysis
1. Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.
2. Retrieve the explained variance to determine how much information can be attributed to each principal component and then answer the following question in your notebook:
• What is the total explained variance of the three principal components?
3. Create a new DataFrame with the PCA data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.
• The first five rows of the PCA DataFrame should appear as follows:

Find the Best Value for k Using the PCA Data
Use the elbow method on the PCA data to find the best value for k using the following steps:
1. Create a list with the number of k-values from 1 to 11.
2. Create an empty list to store the inertia values.
3. Create a for loop to compute the inertia with each possible value of k.
4. Create a dictionary with the data to plot the elbow curve.
5. Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
6. Answer the following questions in your notebook:
• What is the best value for k when using the PCA data?
• Does it differ from the best k-value found using the original data?
Cluster Cryptocurrencies with K-Means Using the PCA Data
Use the following steps to cluster the cryptocurrencies for the best value for k on the PCA data:
1. Initialize the K-means model with the best value for k.
2. Create an instance of K-means, define the number of clusters based on the best value of k, and then fit the model using the PCA data.
3. Predict the clusters to group the cryptocurrencies using the PCA data.
4. Create a copy of the DataFrame with the PCA data and add a new column to store the predicted clusters.
5. Create a scatte rplot using pandas’ plot as follows:
• Set the x-axis as "PC1" and the y-axis as "PC2".
6. Answer the following question:
• What is the impact of using fewer features to cluster the data using K-Means?
Determine the Weights of Each Feature on Each Principal Component
1. Create a DataFrame that shows the weights of each feature (column) for each principal component by using the columns from the original scaled DataFrame as the index.
2. Which features have the strongest positive or negative influence on each component?  
Requirements  
Find the Best Value for k Using the Original Scaled DataFrame (15 points)  
To receive all points, you must:  
• Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11. (5 points)  
• Visually identify the optimal value for k by plotting a line chart of all the inertia values computed with the different values of k. (5 points)  
• Answer the following question: What’s the best value for k? (5 points)  
Cluster Cryptocurrencies with K-Means Using the Original Scaled Data (10 points)  
To receive all points, you must:  
• Initialize the K-means model with four clusters by using the best value for k. (1 point)  
• Fit the K-means model by using the original data. (1 point)  
• Predict the clusters for grouping the cryptocurrencies by using the original data. Review the resulting array of cluster values. (3 points)  
• Create a copy of the original data, and then add a new column of the predicted clusters. (1 point)  
• Using pandas’ plot, create a scatter plot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d". (4 points)  
Optimize the Clusters with Principal Component Analysis (10 points)  
To receive all points, you must:  
• Create a PCA model instance, and set n_components=3. (1 point)  
• Use the PCA model to reduce the features to three principal components, then review the first five rows of the DataFrame. (2 points)  
• Get the explained variance to determine how much information can be attributed to each principal component. (2 points)  
• Answer the following question: What’s the total explained variance of the three principal components? (3 points)  
• Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame. (2 points)  
Find the Best Value for k by Using the PCA Data (10 points)  
To receive all points, you must:  

• Code the elbow method algorithm, and use the PCA data to find the best value for k. Use a range from 1 to 11. (2 points)  
• Visually identify the optimal value for k by plotting a line chart of all the inertia values computed with the different values of k. (5 points)  
• Answer the following questions: What’s the best value for k when using the PCA data? Does it differ from the best value for k that you found by using the original data? (3 points)  
Cluster the Cryptocurrencies with K-Means by Using the PCA Data (10 points)  
To receive all points, you must:  
• Initialize the K-means model with four clusters by using the best value for k. (1 point)  
• Fit the K-means model by using the PCA data. (1 point)  
• Predict the clusters for grouping the cryptocurrencies by using the PCA data. Review the resulting array of cluster values. (3 points)  
• Create a copy of the DataFrame with the PCA data, and then add a new column to store the predicted clusters. (1 point)  
• Using pandas’ plot, create a scatter plot by setting x="PC1" and y="PC2". (4 points)  
Determine the Weights of Each Feature on Each Principal Component (15 points)  
To receive all points, you must:  
• Create a DataFrame that shows the weights of each feature (column) for each principal component by using the columns from the original scaled DataFrame as the index. (10 points)  
• Answer the following question: Which features have the strongest positive or negative influence on each component? (5 points)  
Coding Conventions and Formatting (10 points)  
To receive all points, you must:  
• Place imports at the top of the file, just after any module comments and docstrings, and before module globals and constants. (3 points)  
• Name functions and variables with lowercase characters, with words separated by underscores. (2 points)  
• Follow DRY (Don't Repeat Yourself) principles, creating maintainable and reusable code. (3 points)  
• Use concise logic and creative engineering where possible. (2 points)  
Deployment and Submission (10 points)  
To receive all points, you must:  
• Submit a link to a GitHub repository that’s cloned to your local machine and that contains your files. (4 points)  
• Use the command line to add your files to the repository. (3 points)  
• Include appropriate commit messages in your files. (3 points)  
Code Comments (10 points)  
To receive all points, your code must:  
Be well commented with concise, relevant notes that other developers can understand. (10 points)  

