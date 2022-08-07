#  Real Estate Analysis

## **Business Case and Value-- what hypotheses are you trying to prove?**
 
**Our main business use case is to find out what kind of real estate property is best to invest in (has the highest profit over years) for the user, based on all factores (walkability score, crime rate, school ranking, etc.)**

-  We came up with a golden cluster that groups properties which are  profitable, has all of the desirable features, and is located in a secure area. Various latent variables such as crime rate (property/violence), profit, walkability metrics (walk score, bike score, transit score), and zip code ranking measures are used for this. 
 
On top of that, we performed classification to find the likeability factor to invest and finally used regression techniques to predict the price of the property.

## **Data narrative: tell a story to guide an investor or a retiree or just someone interested in buying…**
- Using the Real_estate Dataset, we performed Data cleaning, Data Preparation and eliminated unwanted columns from the dataset by performing feature selection/Importance. Along with this we have also scraped data related to accessibility score, safety and zipcode ranking from the below mentioned websites to enrich the initial dataset, amalgamate it and improve the feature set and visualization to deduce the best model.
 
https://www.walkscore.com/
https://www.bestplaces.net/crime/
https://www.niche.com/places-to-live/

## **Visualizations, of data prep using first data enrichment (add a dataset to base data set)**
- Performed various visualization techniques to understand the data distribution and behavior. Used pairplot, barplot, line plot and pie chart to understand the relationship between features. 
- Tried Autoviz to understand more about features and data distribution.
 
## **Feature Importance - Gini score**
- We performed various feature importance techniques such as Pearson Correlation
,Chi-Squared, Recursive Feature Elimination, Lasso: SelectFromModel, Tree-based: SelectFromModel using Gini score and came up with the required features for our use case.

## **Data Preparation:**
First, we wrangled the data and removed irrelevant features, which contain a lot of NaN. We used different data imputation techniques such as knn Imputer , Iterative Imputer to impute the missing data.
Then we merged with multiple scraped datasets (Crime Rate, Walkability Score, Zip Code Ranking) to get the final enriched dataset for further analysis.

 
## **Feature Transformation:**
We Performed Feature transformation to transform the dataset to the desired format. Like extracting the State and Zipcode from the dataset. We performed a min-max scaler to uniformly scale the data distribution.   We transformed existing features, added new features to the dataset via amalgamations (Crime Rate info, Walkability info, Zip Code Ranking info, etc..).

## **Dimensionality Reduction via PCA**
We performed dimensionality reduction to the dataset to reduce the  no. of dimensions  to 2 for the purpose of visualizing the data distribution.

## **Implement 3 amalgamations:**

The first data set is the initial dataset provided by the professor for the exam.

**For the First Data Enrichment,** we used the Walkability score(Walk Score, Bike Score, Transit Score)

By scraping the dataset from https://www.walkscore.com/  to the necessary scores for each Zillow property in the dataset.

**How many did you scrape?**We were able to scrap Walkability info for all properties in the initial dataset.

**Second Data Enrichment**we used the Crime score(Violent Crime, Property Crime)  for each locality like zipcode.

By scraping the dataset from https://www.bestplaces.net/crime/  to get the necessary scores for each Zillow property in the first dataset.

**How many did you scrape?** We were able to scrap crime info for all zip codes extracted from the dataset


**Third Data enrichment**-- we used the Zip Code Ranking(School Ranking, Family Ranking, Diversity Ranking and liveability Ranking)  for each Zip Code.

By scraping the dataset from https://www.niche.com/places-to-live/  to get the necessary school ranking around each Zillow property in the first dataset.

**How many did you scrape?** We were able to scrap School Ranking info for all zip codes extracted from the dataset.
 
 
## **Golden Cluster: (Define a Golden cluster and use Fractal Clustering to find it based on the business case you formulate)**
https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=9KlJtxp7bntV
 
Our golden cluster meets the below criteria, where it has properties that are safe and yields high profit , good walkability score and reasonable zip code rank for living.
 
​​During the First Analysis, We ran Kmeans on the entire features of the dataset and used elbow method and cluster metrics like: silhouette_score, calinski_harabasz_score, davies_bouldin_score to understand  the clusters created from the dataset. 
 
With the method described above, we were unable to reach any conclusions. We felt that performing fractal clustering with different objective functions and finding the golder cluster that solves our marketing problem would be one method to come up with a solution for our use case.
 
We clustered based on Walkability in the first Kmeans iteration to select clusters that included properties with greater walkability scores.
 
We clustered using Crime Rate on Top of Walkability Cluster in the Second Kmeans Iteration to arrive at clusters that are extremely secure and have a high walk score.
 
We clustered using the Profit feature (Zestimate - Price) on top of the aforementioned iterated cluster in the last Kmeans iteration to achieve clusters with high Profit, secure, and walkability scores. Which is our analysis' key performance indicator.
 
Finally we arrived at our golder Cluster with around 1.5K properties related to our use case. 
 
## **Classification:**
**https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=WXt_ytp8GngP**
We performed classification on the golden cluster based on likeability (Investment criteria) factors. Here the model classifies whether to invest on the property or not based on the criteria provided by the user. 
Initially, feature extraction is performed to come up with an ‘Invest’ feature based on user likeability criteria. 
 
In order to get the features that influence the invest feature, we performed feature selection using techniques such as Chi-Squared, Recursive Feature Elimination, Lasso: SelectFromModel, Tree-based: SelectFromModel, Gini Index, heatmap and correlation.
 
Finally, we used Muller- loop that comprises of different classification techniques such as Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, Neural Net, Naive Bayes,QDA to classify if the property is worth investing or not based on the features obtained from feature selection and Investment criteria
 
## **Regression:**
**https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=WXt_ytp8GngP**
 
We use several regression techniques to predict the profit one can earn from each property based on the given factors (Walkability, School Ranking around the neighborhood, etc), and use Muller Loop to find out the best regression technique for the case based on r2 Score, Mean Squared Error, and Squared Root Mean Error.
Regression Techniques Used:
Linear Regression, Random Forest Regressor, KNN Regression, MLPRegressor, Decision Tree Regression, AdaBoost Regression, Gaussian Regression, Ridge Regression

## **Implement ml algorithms to build models and refine your data narrative:**
**Classification : **
 
we used Muller- loop that comprises of different classification techniques such as Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, Neural Net, Naive Bayes,QDA to classify if the property is worth investing or not based on the features obtained from feature selection and likeability criteria.
 
**Regression :**
 
we used Muller- loop that comprises of different regression techniques such as Linear Regression, Random Forest Regressor, KNN Regression, MLPRegressor, Decision Tree Regression, AdaBoost Regression, Gaussian Regression, Ridge Regression to predict the property price based on the features obtained from feature selection techniques.

## **Compare relevant tasks in the same table:**
We have a dataframe that captures the performance metric for different algorithms for classification and regression use cases.

**Write a data narrative to interpret the results of each algorithm**
 
**Classification** : **https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=TQXANAsV2B3U&line=1&uniqifier=1**
 
Since we refined our data to meet our business requirements. All Classification algorithms performed well on most occasions.
 
**Analysis of different classification algorithms:**
 
Decision Tree /Random forest(Multiple decision trees) provided a better performance for our investment criteria. Since it was able to understand the features in the come up with conclusions, One aspect of the decision tree is that it memorizes the info in the data and may have the ability to overfit for different validation cases which do not meet our  required criteria. Mostly, the Perfect Decision tree always performs better on training data and mostly has high variance  in the validation data.
 
Knn Algorithms are pretty simple and provide an easy approach to solve classification problems. Since our domain features used in our  uscases are correlated , it was able to choose classes easily since the data has high separability. The core of this classifier depends mainly on measuring the distance or similarity between the tested examples and the training examples.
Its performance was pretty good considering its simplicity for our classification criteria.
 
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. It is based on Bayes theorem.
With this approach we can find the probability of our investment criteria with all the dependent features in the dataset and come up with conclusions. It provided good classification metrics.
Different Performance metrics  are used  for our domain and were visualized for  further understanding.
 
**Regression:**
**https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=CfiysxDwWQ9G&line=1&uniqifier=1**

Linear Regression performed better in predicting the property price based on the important features. Because it  predicts the value in the continuous range instead of classifying the values into categories.
It was able to draw a best fit line since there was a strong correlation with the dependent features in the dataset. 

When the features are highly correlated with the predicted variable, Ridge Regression performed effectively. It provides generalization to our model  to avoid overfitting and underfitting(Low Bias and Low Variance). Our dataset features have high correlation in our price prediction and were able to perform well and come up with a good regression line to satisfy our use case similar to linear regression. When the dataset has multicollinearity then Ridge regression can come to rescue for our problem domain.
 
## **Suggest Latent Variables or Latent Manifolds:**
Walk_Score
Transit_Score
Bike_Score
Violent Crime
Property Crime
Zip_code_Rank_to_live
Zip_code_Rank_Families
Zip_code_Rank_schools
Zip_code_Rank_Diverse
possible profit(zestimate-price)
Adding these features to the initial dataset models accuracy increased and performed better in clustering properties, predicting price of properties and classifying if the property is worth investment or not.

**use appropriate metrics for measuring models and compare them in a table: regression metrics and/or classification metrics (confusion matrix, f1)**
**https://colab.research.google.com/drive/1sbd9-TjMl01Ew_NpJg8-aFfMsgpdWv-e#scrollTo=DxDIHUlu2DL3&line=1&uniqifier=1**
