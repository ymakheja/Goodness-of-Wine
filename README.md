# Predicting goodness of Wine using ridge regression

Implemented Ridge Regression with Leave-One-Out-Cross-Validation error for predicting the goodness points of a wine.


# Data:
The data was scraped from WineEnthusiast during the week of June 15th, 2017 and can be found on the analytics website Kaggle `https://www.kaggle.com/zynicide/wine-reviews`. This dataset provides a variety of features, the points, description (review), variety, country, province etc. For this project we used the points and description features. Points are ratings within range 1-100. However, the dataset contains description for wines rated in range 80-100. Feature vectors are already extracted for each review.


File | Description
---- | -----------
trainData.txt | **Training data matrix for predicting number of stars**
trainLabels.txt | **Training labels, list of the number of stars for each review**
valData.txt | **Validation data**
valLabels.txt | **Validation labels**
testData.txt | **Test data**
featureTypes.txt | **List of features used**

