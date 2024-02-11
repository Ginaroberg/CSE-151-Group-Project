# CSE-151-Group-Project

## Data Preprocessing

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

### other data processing


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

## Data Visualizations 
- Created histogram with average company ratings (average between career_opportunities_rating, comp_and_benefits_rating, culture_and_values_rating, senior_management_rating, work_life_balance_rating)
- Created histogram with company rating to compare with average company rating
- Created bar grapb with distribution of employment types
- Created scatterplot to compare average company rating to salary average estimate
- Created box plot to see salary distribution across the different sectors


