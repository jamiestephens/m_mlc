## Binary Classification: Building a Bankruptcy Predictor Through Random Forest Classification

Using a five-year dataset, I evaluated financial statement metrics to predict 

### Design

This project's business use case emphasizes outside investors or banks who 


### Data

The data was sourced from UCI's [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data), and was originally collected as part of a study through Poland's Wrocław University of Science and Technology. Individual data points are numerical and show financial ratios, with the final variable representing either bankruptcy for the company (1) or no bankruptcy (0). The dataset is 43,405 rows, with 41,314 records showing companies that did not go bankrupt, and 2,091 that did. One highlight feature, that showed the greatest feature importance when tested with scikit-learn's model.feature_importances functionality, was 

<i>Preprocessing</i>

The data was originally available in five separate arff files. I collated these into one dataframe after downloading. There were also enough instances of missing values that it made sense to implement scikit-learn's SimpleImputer. 


### Algorithms

<i>Feature Importance</i>

<i>Models</i>

K-Nearest Neighbors, Logistic Regression, and Gradient Boosting classifiers were tested before random forest was determined to be the model with the most relevance for this business use case. 

<i>Model Evaluation</i>



### Tools

  * Scipy.io, NumPy, and Pandas for data import and manipulation

  * Scikit-learn for modeling

  * Matplotlib for visualizations


### Communication

A confusion matrix and feature importance barchart were presented as part of a slide presentation:
![image](https://user-images.githubusercontent.com/71529189/121619451-a037fa80-ca36-11eb-919f-2dc2508b9b47.png)
