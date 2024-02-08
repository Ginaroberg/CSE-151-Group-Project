# CSE-151-Group-Project

## Data Preprocessing

Converted the values in the following columns to integers 

- company_founded

Converted the values in the following columns to floats 
- salary_avg_estimate

Converted the values in the following columns to lists
- job_description

Convert Salary Estimate period to years and map it so salary_avg_estimate
-salary_avg_estimate and salary_estimate_payperiod and drop salary_estimate payeriod + rename salary_avg_estime

One hot encode
- sector,company_size,revenue

Drop 
- industry because we are going to use sector instead and too many values
- sector,company_size,revenue (after one hot encoding for predictive tasks)
