## Minimum Viable Product


Five years' worth of company records were collected and scaled using sklearn's MinMaxScaler. Initial data exploration was done in eda.py, the preprocessing was completed in preprocessing.py, the feature selection completed in featureselection.py, and initial decision tree work done in decisiontree.py. 
The total number of records was 43,405. Missing values were replaced with 0 in the preprocessing stage, after scaling. 

![image](https://user-images.githubusercontent.com/71529189/121260334-31647100-c87f-11eb-84a5-bcb4a42f4d5b.png)

| Feature Number | Feature                                     | Importance |
|----------------|---------------------------------------------|------------|
| 32             | (Current Liabilities * 365) / Cost of Products Sold | 0.155      |
| 31             | (Gross Profit + Interest) / Sales  | 0.07285    |
| 25             | (Equity - Share Capital) / Total Assets  | 0.0713     |
| 44             | 	(Receivables * 365) / Sales                | 0.06753    |
| 33             | Operating Expenses / Short Term Liabilities      | 0.03411    |

With an initial decision tree analysis the accuracy found was 93%. 
