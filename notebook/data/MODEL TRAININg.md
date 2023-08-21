<span style="font-size: 35pt;">Model Training</span>


<span style="font-size: 16pt;">Import Data and Required Packages</span>

<span style="font-size: 16pt;">Importing Pandas,Numpy,Matplotlib,Seaborn and Warings Library</span>


```python
#Basic Import

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##Modelling

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
```

<span style="font-size: 16pt;">Import the CSV Data as Pandas DataFrame</span>


```python
df=pd.read_csv('stud.csv')
```

<span style="font-size: 16pt;">Show Top 5 Records</span>


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



<span style="font-size: 16pt;">Preparing X and Y variables</span>


```python
X=df.drop(columns=['math score'],axis=1)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
y=df['math score']
```


```python
y.head()
```




    0    72
    1    69
    2    90
    3    47
    4    76
    Name: math score, dtype: int64




```python
## create Column Transformer with 3 types of transformers

num_features=X.select_dtypes(exclude="object").columns
cat_features=X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

num_transformer=StandardScaler()
oh_transformer=OneHotEncoder()

preproccesor=ColumnTransformer(
    [
        ("OneHotEncoder",oh_transformer,cat_features),
        ("standardScaler",num_transformer,num_features)
    ]
)
```


```python
X=preproccesor.fit_transform(X)
```


```python
X
```




    array([[ 1.        ,  0.        ,  0.        , ...,  1.        ,
             0.19399858,  0.39149181],
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
             1.42747598,  1.31326868],
           [ 1.        ,  0.        ,  0.        , ...,  1.        ,
             1.77010859,  1.64247471],
           ...,
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.12547206, -0.20107904],
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.60515772,  0.58901542],
           [ 1.        ,  0.        ,  0.        , ...,  1.        ,
             1.15336989,  1.18158627]])




```python
X.shape
```




    (1000, 19)




```python
y.shape
```




    (1000,)




```python
#separate dataset into train adnd test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
```


```python
X_train.shape,X_test.shape,y_train.shape,y_test.shape
```




    ((800, 19), (200, 19), (800,), (200,))



<font size="5">Create an Evalute function to give all metrics after model Training</font>



```python
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))  # Corrected RMSE calculation
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

```


```python
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoost Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list = []

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)  # Train Model

    # Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate Train and Test dataset
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Performance for Training set')
    print("-Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("-Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("-R2 Score: {:.4f}".format(model_train_r2))
    print('...........................................')

    print('Model Performance for Test set')
    print("-Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("-Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("-R2 Score: {:.4f}".format(model_test_r2))

    r2_list.append(model_test_r2)
    print('=' * 35)
    print('\n')

```

    Linear Regression
    Model Performance for Training set
    -Root Mean Squared Error: 5.3243
    -Mean Absolute Error: 4.2671
    -R2 Score: 0.8743
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 5.3960
    -Mean Absolute Error: 4.2158
    -R2 Score: 0.8803
    ===================================
    
    
    Lasso
    Model Performance for Training set
    -Root Mean Squared Error: 6.5938
    -Mean Absolute Error: 5.2063
    -R2 Score: 0.8071
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 6.5197
    -Mean Absolute Error: 5.1579
    -R2 Score: 0.8253
    ===================================
    
    
    Ridge
    Model Performance for Training set
    -Root Mean Squared Error: 5.3233
    -Mean Absolute Error: 4.2650
    -R2 Score: 0.8743
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 5.3904
    -Mean Absolute Error: 4.2111
    -R2 Score: 0.8806
    ===================================
    
    
    K-Neighbors Regressor
    Model Performance for Training set
    -Root Mean Squared Error: 5.7077
    -Mean Absolute Error: 4.5167
    -R2 Score: 0.8555
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 7.2530
    -Mean Absolute Error: 5.6210
    -R2 Score: 0.7838
    ===================================
    
    
    Decision Tree
    Model Performance for Training set
    -Root Mean Squared Error: 0.2795
    -Mean Absolute Error: 0.0187
    -R2 Score: 0.9997
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 8.2256
    -Mean Absolute Error: 6.5300
    -R2 Score: 0.7220
    ===================================
    
    
    Random Forest Regressor
    Model Performance for Training set
    -Root Mean Squared Error: 2.2813
    -Mean Absolute Error: 1.8088
    -R2 Score: 0.9769
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 5.9927
    -Mean Absolute Error: 4.6196
    -R2 Score: 0.8524
    ===================================
    
    
    XGBRegressor
    Model Performance for Training set
    -Root Mean Squared Error: 0.9087
    -Mean Absolute Error: 0.6148
    -R2 Score: 0.9963
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 6.5889
    -Mean Absolute Error: 5.0844
    -R2 Score: 0.8216
    ===================================
    
    
    CatBoost Regressor
    Model Performance for Training set
    -Root Mean Squared Error: 3.0427
    -Mean Absolute Error: 2.4054
    -R2 Score: 0.9589
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 6.0086
    -Mean Absolute Error: 4.6125
    -R2 Score: 0.8516
    ===================================
    
    
    AdaBoost Regressor
    Model Performance for Training set
    -Root Mean Squared Error: 5.8515
    -Mean Absolute Error: 4.7869
    -R2 Score: 0.8481
    ...........................................
    Model Performance for Test set
    -Root Mean Squared Error: 6.1604
    -Mean Absolute Error: 4.8123
    -R2 Score: 0.8440
    ===================================
    
    
    

<font size="5">Results</font>



```python
pd.DataFrame(list(zip(model_list,r2_list)),columns=['model Name','R2 Score']).sort_values(by=["R2 Score"],ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model Name</th>
      <th>R2 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Ridge</td>
      <td>0.880593</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.880345</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest Regressor</td>
      <td>0.852416</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CatBoost Regressor</td>
      <td>0.851632</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdaBoost Regressor</td>
      <td>0.844043</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lasso</td>
      <td>0.825320</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBRegressor</td>
      <td>0.821589</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K-Neighbors Regressor</td>
      <td>0.783813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decision Tree</td>
      <td>0.721951</td>
    </tr>
  </tbody>
</table>
</div>



<font size="5">Linear Regression</font>


```python
lin_model=LinearRegression(fit_intercept=True)
lin_model=lin_model.fit(X_train,y_train)
y_pred=lin_model.predict(X_test)
score=r2_score(y_test,y_pred)*100
print("Accuracy of the model is %.2f" %score)
```

    Accuracy of the model is 88.03
    

<font size="5">Plot y_pred and y_test</font>


```python
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('predicted')
```




    Text(0, 0.5, 'predicted')




    
![png](MODEL%20TRAININg_files/MODEL%20TRAININg_28_1.png)
    



```python
sns.regplot(x=y_test,y=y_pred,ci=None,color='red')
```




    <Axes: xlabel='math score'>




    
![png](MODEL%20TRAININg_files/MODEL%20TRAININg_29_1.png)
    


<font size="5">Difference between Actual and Predicted Values</font>


```python
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'diffrence':y_test-y_pred})
pred_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual Value</th>
      <th>Predicted Value</th>
      <th>diffrence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>521</th>
      <td>91</td>
      <td>76.507812</td>
      <td>14.492188</td>
    </tr>
    <tr>
      <th>737</th>
      <td>53</td>
      <td>58.953125</td>
      <td>-5.953125</td>
    </tr>
    <tr>
      <th>740</th>
      <td>80</td>
      <td>76.960938</td>
      <td>3.039062</td>
    </tr>
    <tr>
      <th>660</th>
      <td>74</td>
      <td>76.757812</td>
      <td>-2.757812</td>
    </tr>
    <tr>
      <th>411</th>
      <td>84</td>
      <td>87.539062</td>
      <td>-3.539062</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>408</th>
      <td>52</td>
      <td>43.546875</td>
      <td>8.453125</td>
    </tr>
    <tr>
      <th>332</th>
      <td>62</td>
      <td>62.031250</td>
      <td>-0.031250</td>
    </tr>
    <tr>
      <th>208</th>
      <td>74</td>
      <td>67.976562</td>
      <td>6.023438</td>
    </tr>
    <tr>
      <th>613</th>
      <td>65</td>
      <td>67.132812</td>
      <td>-2.132812</td>
    </tr>
    <tr>
      <th>78</th>
      <td>61</td>
      <td>62.492188</td>
      <td>-1.492188</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 3 columns</p>
</div>




```python

```
