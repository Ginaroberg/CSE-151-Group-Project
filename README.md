# CSE-151-Group-Project

# Introduction
In today's competitive job market, having insights into salary ranges is essential for informed career decisions. Glassdoor provides extensive data on job listings, salaries, and companies. Using the Glassdoor Data Science Jobs - 2024 dataset, we aimed to find patterns between job attributes and salary ranges. This dataset includes job requirements, company ratings, salary, and more. This project offers insights for job seekers, empowering them to make informed career decisions. Additionally, it has the potential to reveal hidden trends and address issues like wage inequality. A good predictive model for salary ranges can benefit people by providing clarity on salary expectations and helping employers provide competitive compensation to attract talent. Overall, this project aims to foster transparency and efficiency in the labor market, benefiting individuals and organizations in the data science industry.

# Methods

## Data Exploration
In our data exploration, we delved into various aspects of the Glassdoor Data Science Jobs - 2024 dataset to gain insights into job listings, company attributes, and salary estimates. ![Image Alt Text](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/Distribution_of_average_company_rating.jpg)

 We began by visualizing the distribution of average company ratings, computed as the average of ratings related to career opportunities, compensation and benefits, culture and values, senior management, and work-life balance. This histogram provided a comprehensive overview of company ratings, enabling us to identify trends and outliers in employer satisfaction. The histogram shows that the data has an overall normal distribution, slightly skewed left with most average ratings between 3.5 and 4.0. Notably, there are more ratings on the right side of the mode compared to the left side. This distribution indicates that a significant proportion of companies in the dataset are perceived favorably by employees, with ratings clustered around the higher end of the scale

  ![Image Alt Text](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/Distribution_of_company_rating.jpg)

  Additionally,  we created histogram with company rating to compare with average company rating to gauge any disparities or correlations between these metrics.  Unlike the histogram for average company ratings, this histogram does not exhibit a normal distribution. However, it is evident that the majority of ratings still fall within the range of 3.5 to 4.0, which aligns with our observations from the previous analysis. Interestingly, the distribution of ratings remains skewed left, indicating that most companies tend to receive favorable ratings, with fewer instances of lower ratings.
  
Furthermore, we examined the distribution of employment types across job listings, offering insights into the prevalence of full-time, part-time, and contract positions. Moreover, we investigated the relationship between average company rating and salary average estimate through a scatterplot, aiming to discern any associations between employer reputation and compensation levels. Lastly, we utilized a box plot to visualize the salary distribution across different sectors, allowing for comparisons and identification of sectors with higher salary ranges. Through these data visualizations, we gained valuable insights into the job market landscape, providing a foundation for further analysis and model development.


## Preprocessing
In the data preprocessing phase, we focused on preparing the dataset for analysis by converting values into appropriate formats, handling missing data, and performing feature engineering. Firstly, we converted the "company_founded" column to integers and the "salary_avg_estimate" column to floats to ensure numerical accuracy. Additionally, we transformed the "job_description" column into lists and created a new column "salary_estimate_per_year" by converting salary estimates to yearly values based on the "salary_estimate_payperiod" factor. To address missing data, we imputed values using various strategies. For the "company" column, we filled in missing entries with "unknown" to maintain completeness. Similarly, we imputed missing values in the "company_size" column with "unknown" and replaced NaNs in the "company_founded" column with "0000.0" for consistency. For categorical columns like "employment_type," "industry," "sector," and "revenue," we used "unknown" as a placeholder for missing values. To handle missing ratings for companies, we computed the average of relevant rating columns such as 'career_opportunities_rating,' 'comp_and_benefits_rating,' 'culture_and_values_rating,' 'senior_management_rating,' and 'work_life_balance_rating.' This approach ensured that missing company ratings were replaced with reasonable estimates based on other available data. For the "salary_avg_estimate" column, we employed k-nearest neighbors (KNN) imputation to fill in missing salary values, leveraging the similarity between instances to infer missing values more accurately. Additionally, we imputed the mode of the "salary_estimate_payperiod" column to determine the most common pay period, whether yearly, monthly, or hourly. Overall, our data preprocessing efforts involved converting data types, handling missing values through imputation, and performing feature engineering to prepare the dataset for further analysis, ensuring its integrity and reliability in subsequent modeling tasks.

df.head()


Model 1
Our first model was a RandomForestClassifier. We used the following features to classify salary_range: 
 - 'company_rating'
 - 'company_founded'
 - 'career_opportunities_rating'
 - 'comp_and_benefits_rating'      
 - 'culture_and_values_rating'     
 - 'senior_management_rating'       
 - 'work_life_balance_rating'
We scaled the data using MinMaxScaler().  We used a parameter of n_estimators=100.
Model 2
Our second model was a Decision Tree Classifier. We used the following features to classify the salary_range:
 - 'company_rating'
 - 'company_founded'
 - 'career_opportunities_rating'
 - 'comp_and_benefits_rating'      
 - 'culture_and_values_rating'     
 - 'senior_management_rating'       
 - 'work_life_balance_rating'
Using these features, we hyperparameter turned the following parameters for the decision tree: 'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2]

Model 3
Our third model was a XGboost. We used the following features to classify the salary_range:
 - 'company_rating'
 - 'company_founded'
 - 'career_opportunities_rating'
 - 'comp_and_benefits_rating'      
 - 'culture_and_values_rating'     
 - 'senior_management_rating'       
 - 'work_life_balance_rating'

For XgBoost we first started with a learning rate of .05 as well as default parameters. We then ran grid search cv to find the optimal parameters to reduce overfitting. After running grid search cv for learning rate, max_depth, n_estimamtors, subsample, and colsampe_bytree with cv =5 and scoring = accuracy we were able to find the best parameters which were {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}

# Resutlts
# Model 1: Random Forest Classifier
For our first model, we arbitrarily chose to use a random forest classfier because we wanted to gauge how a model would initially perform for our classification task.  We used the following features to predict salary_range: 
 - 'company_rating'
 - 'company_founded'
 - 'career_opportunities_rating'
 - 'comp_and_benefits_rating'      
 - 'culture_and_values_rating'     
 - 'senior_management_rating'       
 - 'work_life_balance_rating'

![features_confusion_matrix](https://github.com/Ginaroberg/CSE-151A-Group-Project/assets/94018260/3257be36-3988-4e93-9c81-722d92a3dc4d)

With this we were able to achieve following metrics:

Training Accuracy: 0.908289241622575
Training Mean Squared Error: 0.4497354497354497

Validation Accuracy: 0.5352112676056338
Validation Mean Squared Error: 1.9507042253521127

Testing Accuracy: 0.6797752808988764
Testing Mean Squared Error: 1.1123595505617978

![results_confusion_matrix](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/Confusion_Matrix_RFC.jpg)

## Fitting Graph

![fitting_graph_rfc](https://github.com/Ginaroberg/CSE-151A-Group-Project/assets/94018260/8cd670cf-364e-4599-9bb6-8db1bbc272ec)

Based on the fitting graph, we can see that the model may be overfitting to the training data.  Because the accuracy for the training data is drastically higher than the accuracy achieved by the accuracy of the validation data, and stays this way when the complexity of the model increases, this indicates overfitting.

## Random Forest Classifier Conclusion
In conclusion we found that our first model using Random Forest Classifier is overfitting to our training data.  To improve our model, we could further experiment with changing hyperparameters, including max_depth, and min_samples_leaf.  Another approach we could implement would be cross validation to improve our model as well.

# Future Models
The next two models we would try are SVM and neural networks.  Our reasoning behind trying SVM in our next model is because it can handle non-linear decision boundaries.  Based on our confusion matrix, many of the features do not have a strong correlation with the 'salary_avg_estimate_per_year' and 'salary_range' columns.  As a result, this indicates a non linear relationship salary has with the other features.  SVM could potentially handle this better and produce better predictions.  Our reasoning for attempting neural networks in our next model is also similar, since neural networks can also handle nonlinear relationships.  Another reason we want to attempt using neural networks is because they can utilize other datatypes.  Our dataset contains text data, which we want to try using to improve our salary predictions.

# Model 2: Decision Tree Classifier
In our second model, we decided to attempt using a decision tree classifier.  We used the same features to predict salary_range as we did in our random forest classifier. To refine the performance of our model, we performed hyper parameter tuning where we used grid search cv with the parameters for decision tree which were criterion, splitter, max_depth, and min_samples_split. Based on the results, we were able to find our best model.

With this model we were able to achieve the following metrics:

Training Accuracy: 0.9065255731922398
Training Mean Squared Error: 0.4514991181657848

Validation Accuracy: 0.5774647887323944
Validation Mean Squared Error: 1.8943661971830985

Testing Accuracy: 0.702247191011236
Testing Mean Squared Error: 1.0337078651685394

# Fitting Graph
Based on the results of the model and the fitting graph, our model is still overfitting on the training data.

![fitting_graph_rfc](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/decision_tree.jpg)

# Decision Tree Classifier Conclusion
With this model, we were able to achieve a higher accuracy.  The model is not overfitting as much, but there is still a significant difference between the training and testing mean-squared error.

# Future Models 
The next model we plan to try is neural networks.  This is our next approach because it can possibly utilize the other types of data from our dataset. 

# Model 3: XG Boost

Our third model was XG Boost. Our baseline model only included learning rate as a parameter, we ended up using .05 as our learning rate and our baseline model achieved an accuracy score of .3802 and an accuracy of .90 on our training data.  We then created our hyperparameter tuned model and achieved an accuracy score of the test dataset of .623 and an accuracy score of .750 on our training dataset.
![fitting_graph_rfc](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/xgb_fitting.png)

# Discussion


# Conclusion 

# Collaboration
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

One of the biggest aspects we could have done differently was finding different ways to incorporate all the data we had in our dataset.  An aspect that we could have done differently was implementing TF-IDF for the job_descriptions.  When we were performing our data preprocessing, we were trying to convert our data to be in the most usable form, like one-hot encoding categorical values.  We wanted to find a way to implement the job_description column.  Since it was text data, we thought we could perform sentiment analysis using TF-IDF.  However, we face difficulties with implementing it without adding on hundreds of columns. Since each text description had multiple words, we were unsure of how to relate the value to the salary range and use it as a feature.  We could have tried to implement the other columns that contained data of non-numerical form as a feature for our model. Additionally, we continuously used the same columns for all of our models.  We could have experimented and tried to implement more variety or refined what we used more. Overall, we were thorough in examining our models and their performance, but there are aspects we have learned that we can keep in mind for future projects.




# Link to Jupyter Notebook
[Open Jupyter Notebook](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/CSE%20151A%20Group%20Project%20Notebook.ipynb)





