# CSE-151-Group-Project

## Data Preprocessing
Converted the values in the following columns to integers 
- company_founded

Converted the values in the following columns to floats 
- salary_avg_estimate

Converted the values in the following columns to lists
- job_description

Create salary_estimate_per_year column 
- convert salary_estimate_payperiod into factors of year
- multiply salary_avg_estimate by salary_estimate_payperiod
- drop salary_estimate payeriod + rename salary_avg_estimate to salary_avg_estimate_per_year

One Hot Encode the following columns 
- company_size
- revenue
- sector

Drop industry column

Tokenized job_description and performed sentiment analysis with TF-IDF

### Data Imputation
Almost every column in the dataframe has missing data. We decided to impute missing data in certain ways depending on the column
- company column: inpute "unknown"
- company_rating column: imputed the average of 'career_opportunities_rating','comp_and_benefits_rating', 'culture_and_values_rating','senior_management_rating',       
'work_life_balance_rating' columns.
- job_description column: dropped the 12 rows of missing data in job_description as we are going to use job description for text analysis.
- company_size column: impute 'unknown'
- company_founded column: impute (0000.0) for the missing years
- employment_type, industry,sector,revenue columns: imputed 'unknown'
- company related ratings (ex. 'career_opportunities_rating','comp_and_benefits_rating'): impute the average of each column.
- salary_avg_estimate column: imputed values based on knn imputation for salary.
- salary_estimate_payperiod column: imputed mode of the column as it was diffcult to assess wheter or not salary was yearly,monthly, or hourly

## Data Visualizations 
- Created histogram with average company ratings (average between career_opportunities_rating, comp_and_benefits_rating, culture_and_values_rating, senior_management_rating, work_life_balance_rating)
- Created histogram with company rating to compare with average company rating
- Created bar grapb with distribution of employment types
- Created scatterplot to compare average company rating to salary average estimate
- Created box plot to see salary distribution across the different sectors

# Predictive Task 
Our goal is to predict the range that a company's average estimated annual salary will fall into, based on various features.  We want to discover what features will have the highest correlation with the average estimated annual salary.  This means we are performing classification, based on the 8 differen salary ranges, that we created based on the average estimated annual salaries from the data.

# First Model: Random Forest Classifier
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

![results_confusion_matrix](https://github.com/Ginaroberg/CSE-151A-Group-Project/assets/94018260/16a13af3-a580-4748-bec8-f22498968842)

# Fitting Graph

![fitting_graph_rfc](https://github.com/Ginaroberg/CSE-151A-Group-Project/assets/94018260/8cd670cf-364e-4599-9bb6-8db1bbc272ec)

Based on the fitting graph, we can see that the model may be overfitting to the training data.  Because the accuracy for the training data is drastically higher than the accuracy achieved by the accuracy of the validation data, and stays this way when the complexity of the model increases, this indicates overfitting.

# Random Forest Classifier Conclusion
In conclusion we found that our first model using Random Forest Classifier is overfitting to our training data.  To improve our model, we could further experiment with changing hyperparameters, including max_depth, and min_samples_leaf.  Another approach we could implement would be cross validation to improve our model as well.

# Future Models
The next two models we would try are SVM and neural networks.  Our reasoning behind trying SVM in our next model is because it can handle non-linear decision boundaries.  Based on our confusion matrix, many of the features do not have a strong correlation with the 'salary_avg_estimate_per_year' and 'salary_range' columns.  As a result, this indicates a non linear relationship salary has with the other features.  SVM could potentially handle this better and produce better predictions.  Our reasoning for attempting neural networks in our next model is also similar, since neural networks can also handle nonlinear relationships.  Another reason we want to attempt using neural networks is because they can utilize other datatypes.  Our dataset contains text data, which we want to try using to improve our salary predictions.

# Second Model: Decision Tree Classifier
In our second model, we decided to attempt using a decision tree classifier.  We used the same features to predict salary_range as we did in our random forest classifier. To refine the performance of our model, we performed hyper parameter tuning we used grid search cv. Based on the results, we were able to find our best model.

With this model we were able to achieve the following metrics:

# Fitting Graph
Based on the results of the model and the fitting graph, our model is still overfitting on the training data.

# Decision Tree Classifier Conclusion
With this model, we were able to achieve a higher accuracy.  The model is not overfitting as much, but there is still a significant difference between the training and testing mean-squared error.

# Future Models 
The next model we plan to try is neural networks.  This is our next approach because it can possibly utilize the other types of data from our dataset. 

# Link to Jupyter Notebook
[Open Jupyter Notebook](https://github.com/Ginaroberg/CSE-151A-Group-Project/blob/main/CSE%20151A%20Group%20Project%20Notebook.ipynb)





