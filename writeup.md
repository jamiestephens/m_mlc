## Binary Classification: Building a Bankruptcy Predictor Through Random Forest Classification

Using a five-year dataset, I evaluated records showing company's financial statement ratios to construct a Random Forest classifier that would determine if a company would go bankrupt. I opted to build this from an outside investor's point of view, so interpretability wasn't a top priority. Investors are unlikely to be interested in the exact reason why an algorithm believes a company will go bankrupt, just that it is likely to, and that was a large factor in my decision to use a Random Forest model. I first evaluated the data, and utilized sklearn's SimpleImputer to eliminate missing values before feeding it into the model. The model's success was highly dependent on the type of class weighting solution I used. 

### Design

This project's business use case emphasizes outside investors or banks who might be interested in loaning money to struggling businesses. If a company declares bankruptcy, lenders have a reasonable chance of seeing their money returned, but will be uninterested in taking that chance. For that reason I felt that false positives (predicting a company would go bankrupt incorrectly) were somewhat acceptable in the predictions, but false negatives (predicting a company would stay in business incorrectly) were not. It was also relevant to note that financial statement data is likely to be correlative, not causative - a company is unlikely to point to a poor financial ratio as the reason for filing bankruptcy, but it could very well be a symptom of poor operations. 

This was how ROC (AUC) Score was determined to be the primary metric for model comparisons, as well as a confusion matrix that clearly indicated the number of true positives, false positives, true negatives, and false negatives. 


### Data

The data was sourced from UCI's [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data), and was originally collected as part of a study through Poland's Wrocław University of Science and Technology. Individual data points are numerical and show financial ratios, with the final variable representing either bankruptcy for the company (1) or no bankruptcy (0). The dataset is 43,405 rows, with 41,314 records showing companies that did not go bankrupt, and 2,091 that did. One highlight feature, that showed the greatest feature importance when tested with scikit-learn's model.feature_importances functionality, was profit on operating activities / financial expenses. 

<i>Preprocessing</i>

The data was originally available in five separate arff files. I collated these into one dataframe after downloading. There were also enough instances of missing values that it made sense to implement scikit-learn's SimpleImputer. This class takes the average of a column and uses that to fill any missing values along the column.

Sklearn's MinMaxScaler was implemented for test models that couldn't natively handle data points on different scales, but this was discarded when Random Forest was determined to be the optimal model for this dataset, as decision trees are more robust to numerical instabilities.


### Algorithms

<i>Feature Importance</i>

Some preliminary feature importance was done before modeling. This can be found in featureselection.py. Given the uninterpretable nature of the Random Forest model, it made sense to explore the features briefly to see if any outliers existed. Ultimately it was found that there were few outliers and the Random Forest model would be able to accommodate all of the variables. 

<i>Models</i>

K-Nearest Neighbors, Logistic Regression, and Gradient Boosting classifiers were tested before Random Forest was determined to be the model with the most relevance for this business use case. Code referring to these attempts can be found in the [Unused](https://github.com/jstephens/m_mlc/tree/main/Unused) folder. 

The Random Forest model used had variable class weights (altered as an argument through the constructor). To save computing time and reduce model overfitting, a max_depth of 10 was specified (i.e., the maximum depth of a tree).

<i>Model Evaluation</i>

Several Random Forest models were evaluated for the purposes of this project, with identical hyperparameters except for the class_weighting. Training and test data had a 80:20 split, and a stratified K-fold cross validation score was calculated from the model, with a n_splits value of 5. Models with no class weighting to counteract the fact that 94.5% of the dataset had a binary outcome of 0, predictably, had a higher false negative rate. Indicating a class_weighting of 'balanced' also didn't result in the best outcomes. Assigning a value of 1 to all 0 outcomes and 10 to all 1 outcomes had the best results within the confusion matrix, with the fewest false negatives and false positives given the same test data size. This provides some flexibility in a business setting for investors to accommodate varying levels of risk. 

Unbalanced model
* Mean ROC (AUC) score: 0.912 (standard deviation of 0.023)
* 258 false negatives, 0 false positives

Balanced model
* Mean ROC (AUC) score: 0.916 (standard deviation of 0.015)
* 362 false negatives, 38 false positives

1:10 balancing model
* Mean ROC (AUC) score: 0.922 (standard deviation of 0.017)
* 111 false negatives, 77 false positives

1:20 balancing model
* Mean ROC (AUC) score: 0.915 (standard deviation of 0.017)
* 371 false negatives, 35 false positives

### Tools

  * Scipy.io, NumPy, and Pandas for data import and manipulation

  * Scikit-learn for modeling

  * Matplotlib and scikitplot for visualizations


### Communication

Confusion matrices and a feature importance barchart were presented as part of a slide presentation: 
![image](https://user-images.githubusercontent.com/71529189/121619451-a037fa80-ca36-11eb-919f-2dc2508b9b47.png)

![image](https://user-images.githubusercontent.com/71529189/121652042-43096c80-ca69-11eb-861e-7599b894791a.png)


