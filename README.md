# XGBoost for anonymous data
## What is  it for?
This project was made for analysing anonymous data
Due to the fact that we doesn't know the data we can't expect what factor will be the most informative. As we can see in the file [internship_train.csv][train] in the tabular data we have 53 unknown colomns and colomn 'target'. The last column is responsible for the result of the model. Our task is to obtain the nearest results and with a help of XGBoost. And predict new tagets with [internship_hidden_test.csv][test] file. ```.head()```
## Explanation
-   Let's read [internship_train.csv][train] data and read 'head' of the file using pandas function
-   Separate features data and target data
-   Observe statistics using ```.describe()```
-   Observe correlation matrix using ```.corr()```. Matrix can show that there is weak correlation between features and we can also check it with a help of Principal Components Analysis (PCA):
```sh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
x = StandardScaler().fit_transform(df)
pca = PCA(n_components=.9)
pca.fit(x)
print(pca.explained_variance_)
[1.94138443 1.046112   1.04440732 1.04185583 1.03558478 1.03344805
 1.03173497 1.02984077 1.02773463 1.02737767 1.02538878 1.02247186
 1.02187056 1.01964184 1.01866775 1.01803565 1.01672723 1.01494346
 1.01335583 1.01240745 1.00781751 1.00718217 1.00593544 1.00524473
 1.00411936 1.00206326 1.00165176 1.00024285 0.99762329 0.99692237
 0.99551076 0.99424062 0.99328203 0.99156375 0.9908821  0.98956316
 0.98669645 0.98466061 0.98342996 0.98152507 0.97923085 0.97804753
 0.97688517 0.97397727 0.97295431 0.97026949 0.96966345 0.96693241]
```
Here we can see, if we want to describe 90% of data we need 48 features that is on 5 features less that we used.
-   Use XGBRegressor  with these parametrs:
-   Fit model and predict data for [internship_hidden_test.csv][test]
-   With a help of GridSearch find better values for XGBoost parametrs
-   Compare results with target
-   Save predictions in file [model_predictions.csv][predictions]

## Results
GridSearch found one of the best values for two parametrs: ```min_child_weight``` and ```max_depth```.These parameters have been chosen, in  case of overfitting. Grid search found value 6 for ```max_depth``` and 9 for ```min_child_weight```

After training model the root mean square deviation was = 0.001063
We shouldn't forget that this value is for trained data if we put new data that wasn't in a train sample we can obtain worse results. For now, the results are good for prediction new data.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [train]: <https://github.com/Armkeyter/XGBoost_for_Anonymous_data/internship_train.csv>
   [test]: <https://github.com/Armkeyter/XGBoost_for_Anonymous_data/internship_hidden_test.csv>
   [test]: <https://github.com/Armkeyter/XGBoost_for_Anonymous_data/model_predictions.csv>