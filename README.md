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

With this we were able to achieve following metrics:

Training Accuracy: 0.908289241622575
Training Mean Squared Error: 0.4497354497354497

Validation Accuracy: 0.5352112676056338
Validation Mean Squared Error: 1.9507042253521127

Testing Accuracy: 0.6797752808988764
Testing Mean Squared Error: 1.1123595505617978


4. Where does your model fit in the fitting graph.

5. What are the next 2 models you are thinking of and why?


