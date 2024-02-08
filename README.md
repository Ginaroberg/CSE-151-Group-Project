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
