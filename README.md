- ### Sample Data Set:
- **Data (2016-01-01 to 2017-07-01) sampled at 5 minutes.** 
- **The Y features are the target variables and the X features are the independent variables.** 

Tasks:
1.	Clean and pre-process the data
2.	Summarize the data 
3.	Find anomalies in the target variables (Y1 , Y2). Explain your method and your findings
4.	Forecast the Y variables for next 14 days (2017-07-02 to 2017-07-15). You should focus on improving accuracy of your predictions 
5.	Create a detailed report with plots and explanation of the analysis you’ve done
6.	Submit both the report and the code


# Report


***Prepair data:***
First of all, I need to download the data and get basic information

``` 
data_train = pd.read_csv('dataset_for_technical_assessment.csv')
display(data_train)
data_train.info()
```
 
- first 5 lines and basic information on the number of values, Nuns and data types

- Then I need to find Nan values.
```
data_train.isnull().sum().sort_values(ascending=False)
```
 
**Nan values**

- Then Drop Nans and change the Description column type for DateTime.
```
data_train = data_train.dropna().reset_index(drop=True)

data_train.Description = pd.to_datetime(data_train.Description)
```

- Then I need to find anomalies in the target variables y1 , y2.


- We can notice that y1 and y2 alternate. This means that they alternately measure something by changing places. So they can be combined into one variable and used as a target.

```
data_train['y'] = data_train['y1'] + data_train['y2']
fig = px.line(data_train, x='Description', y="y", title='y1 + y2')
fig.show()
```
 
**After merging, we can see that some values go beyond 1. This is not true because it happens at the time of the change and 2 readings are read.
To fix this, I just take the average value if the sum is greater than 2**
```
def group(row): 
    
    if row['y'] > 1:
        return (row['y1'] + row['y2']) / 2

    return row['y']
data_train['y'] = data_train.apply(group, axis=1) # create a column with the received values
```

- Next, to remove anomalies, I brought out a boxplot.
 
- After that, I need to find strongly correlated features and remove them.
```
plt.figure(figsize=(15, 10)) 

mask = np.triu(np.ones_like(data_train.corr(), dtype=np.bool)) 
heatmap = sns.heatmap(data_train.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG') 
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':25}, pad=20);
```

- x4 and x5 have a strong correlation and x3, x2 too
- For x3, x2 I just took the average and removed them, and in x4 and x5, I decided to simply remove x4 since it has a minimal correlation with the target feature
```
data_train['x23'] = (data_train['x2'] + data_train['x3']) / 2
data_train = data_train.drop(['x2', 'x3'], axis=1)

data_train = data_train.drop('x4', axis=1)
```

- To reduce the number of values and reduce the spread, I combined the time to hours.
```
df = data_train.groupby(pd.Grouper(key="Description", freq="h")).sum()
```
- To begin with, I will use Prophet, but for him I need to prepare the data by dividing it into dates and target
```
df = data_train.groupby(pd.Grouper(key="Description", freq="h")).sum()
df = df.query('y < 11 and y > 7.5')
df = df.rename(columns = {'index':'Description'})
df.reset_index(inplace=True)
# del data
data = df[['Description', 'y']]
data.rename(columns = {'Description' : 'ds'}, inplace = True)
```

- The target variables has missing values in May. I will consider it as a holiday.

```
wrong = pd.DataFrame({
'holiday': 'wrong',
'ds': pd.to_datetime(['2017-05-02', '2017-05-03', '2017-05-04', '2017-05-05', '2017-05-06', '2017-05-07', '2017-05-08']),
'lower_window': -1,
'upper_window': 1,
})
```

### Now we need to train the model. 
```
%%time

wrong = pd.DataFrame({
'holiday': 'wrong',
'ds': pd.to_datetime(['2017-05-02', '2017-05-03', '2017-05-04', '2017-05-05', '2017-05-06', '2017-05-07', '2017-05-08']),
'lower_window': -1,
'upper_window': 1,
})

m = Prophet(holidays = wrong,
            growth="linear", 
            daily_seasonality=True, 
            yearly_seasonality = False,
            weekly_seasonality = False,
            seasonality_mode = "multiplicative",
            changepoint_prior_scale = 0.1,
            seasonality_prior_scale = 35
            )

m.add_seasonality(name="monthly",period=30, fourier_order = 3)
m.add_seasonality(name="weekly",
                                period = 7,
                                fourier_order = 40)
# m.add_seasonality(name="yearly", 
#                                 period=365.25,
#                                 fourier_order = 35)
m.add_seasonality(name="quarter",
                                period=365.25 / 4,
                                fourier_order = 5,
                                prior_scale = 5)
m.add_seasonality(name="1/2year",
                                period=365.25 / 2,
                                fourier_order = 5,
                                prior_scale = 5)
# m.add_country_holidays(country_name = 'US')

m.fit(data)
future = m.make_future_dataframe(periods= 14*24, freq='h',  include_history = False)
forecast1 = m.predict(future)

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

- Prediction results


- Add new values back to the table
```
forecastdf = forecast1[['ds', 'yhat']].rename(columns = {'ds':'Description', 'yhat': 'y'})
data = pd.concat([df, forecastdf])
data
```

- Then I add lags and because of this it will be easy to make predictions. I will also break the date into years, months and hours.  14*24 Because 14 days and 24 hours in 1 day
```
def make_features(data, max_lag, rolling_mean_size):
     # four new calendar attributes: year, month, day and day of the week
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    # lagging values. max_lag, which will set the maximum lag size.
    for i in trange(1, len(df.columns)):
        for lag in range(1, max_lag + 1):
            if df.columns[i] != 'y':
                data['lag_{}'.format(lag)] = data[df.columns[i]].shift(lag)
make_features(data, 14*24, 5)
data
```

- Now I create 3 selections. Testing, validation, and training
```
data_split = data[:-(14*24)]
test = data[-(14*24):]
train, valid = np.split(data.sample(frac=1, random_state=12345),
                                 [int(.85*len(data))])

train = train.dropna()
valid = valid.dropna()
test = test.dropna()
# variables for features and target feature
features_train = train.drop(['y'], axis=1)
target_train = train['y']

features_valid = valid.drop(['y'], axis=1)
target_valid = valid['y']

features_test = test.drop(['y'], axis=1)
target_test = test['y']
print(train.shape)
print(valid.shape)
print(test.shape)
```

- ***Shapes:***
- Train - (11166, 347)
- Valid - (1986, 347)
- Test  - (336, 347)

- I will be using 3 models: Linear Regression, Gradient Boosting, and Random Forest
```
LinearRegression:
model = LinearRegression()
model = model.fit(features_train, target_train)
predictions_valid = model.predict(features_valid)
mse = mean_squared_error(target_valid, predictions_valid)
print("MSE:", mse)
rmse = mse ** 0.5 
print("RMSE:", rmse)
```
```
MSE: 1.1788686255379315
RMSE: 1.085757166929112
Wall time: 470 ms
```


```
Gradient Boosting:
%%time
model1 = CatBoostRegressor(iterations=100, learning_rate=0.5, depth = 8)
# Fit model
model1.fit(features_train, target_train, verbose=10) 
predictions_valid1 = model1.predict(features_valid)
mse1 = mean_squared_error(target_valid, predictions_valid1)
print("MSE:", mse1)
rmse1 = mse1 ** 0.5 
print("RMSE:", rmse1)
```
```
MSE: 0.008945310167465538
RMSE: 0.09457964985907664
Wall time: 31.2 s  
```
```
Random Forest:
%%time
model4 = RandomForestRegressor(n_estimators=100, max_depth = 13)
model4.fit(features_train, target_train) 
predictions_valid4 = model4.predict(features_valid)
mse4 = mean_squared_error(target_valid, predictions_valid4)
print("MSE:", mse4)
rmse4 = mse4 ** 0.5 
print("RMSE:", rmse4)
```
```
MSE: 0.0035735173195037178
RMSE: 0.05977890363249997
Wall time: 4min 58s
```

- I think gradient boosting gives better results so I will use it.
```
%%time
predictions_test = model1.predict(features_test)
mse1 = mean_squared_error(target_test, predictions_test)
print("MSE:", mse1)
rmse1 = mse1 ** 0.5 
print("RMSE:", rmse1)
```
```
MSE: 0.004078315858244003
RMSE: 0.06386169319900627
Wall time: 55.6 ms
 ```
- Prophet and gradient boosting predictions
 
- Last step is final visualization 
```
plt.figure(figsize=(15, 5))
plt.ylim([7.6, 11])
plt.plot(data_split.index, data_split.y, label = "line 1")
plt.plot(fin.index, fin.predictions_test, label= "line 2")
plt.legend()
plt.show()
```
