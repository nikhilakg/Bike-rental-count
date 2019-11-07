# Bike-Rental_Prediction
### Problem Statement 
The objective of this Case is  prediction of bike rental count on daily basis based on the environmental and seasonal settings.

### Data
Our task is to build a regression model that will predict the bike rental on daily basis based on different environmental and seasonal settings. Given below is a sample dataset which we are using to predict the bike rental count.The varibale names are rather self explanatory.

![day_data](https://user-images.githubusercontent.com/20225277/46469294-9cb53580-c7f0-11e8-991c-fc4d40246367.png)

### Pre Processing
Distribution of "cnt"  - The count of daily bikes rented

![correaltion plot](https://user-images.githubusercontent.com/20225277/46470014-d0915a80-c7f2-11e8-99f7-a2ed7c98e636.png)

Distribution of Number of Bikes-Rented by Month

![monthly sales](https://user-images.githubusercontent.com/20225277/46470632-f15aaf80-c7f4-11e8-8483-541c12642af3.png)

Distribution of Number of Bikes-Rented by Season

![season sales](https://user-images.githubusercontent.com/20225277/46470527-945ef980-c7f4-11e8-97fc-a21341b56983.png)

Plot of "temp" -temperature Vs. "cnt" - The count of bikes rented

![rplot-temp vs cnt](https://user-images.githubusercontent.com/20225277/46470855-b73ddd80-c7f5-11e8-9f60-517a50a2c40d.png)

Plot of "dteday" - The dates Vs "cnt" - The count of bikes rented

![rplot -dteday vs cnt](https://user-images.githubusercontent.com/20225277/46470883-d0468e80-c7f5-11e8-8033-56d34dade2f3.png)

##### Outlier Analysis
We find outliers in windseed 
![windspeed](https://user-images.githubusercontent.com/20225277/46470648-00d9f880-c7f5-11e8-8f78-a3535991ee03.png)

### Feature Selection
Based on the correlation plot, we remove atemp.

![correlation plot](https://user-images.githubusercontent.com/20225277/46470825-9bd2d280-c7f5-11e8-9c9a-4c5d5d570f34.png)

### Model Selection
##### Multiple Linear Regression
Implemented multiple linear regression with an accuracy of 87%

##### Random Forest
Implemented Random Forest with an accuracy of 90%

