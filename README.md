# zillow-clustering-project

This repository holds our work and final report for the Clustering project.

---

# About the Project

## Project Goal
The goal of this project is to identify drivers of error in predicting assessed home value for single family properties so that we can become a stronger competitor in the home-buying market by more accurately predicting home values.

## Executive Summary
With a main focus on logerror drivers, we explored the data using visualizations, statistical testing, and clustering. We found `beds`, `baths`, and `sq_ft` to be significant features with which to cluster, and we decided to use them as features in our models as well. After comparing the peformance of four models, we concluded that our best model was an Ordinary Least Squares model utilizing our clusters along with `tax_value`, `age`, and `sq_ft`.

---
## Project Description
The more logerror we have, the more money we stand to lose. We want to find the drivers of Zestimate logerror in an effort to minimize logerror and maximize accuracy in predicting home values. To do this, we will use clustering to identify patterns in our 2017 single-unit property data and use those clusters to build a model which will be used for predicting logerror. If we can predict logerror, we can use those predictions to make more accurate predictions of home values.

## Initial Questions
1. Does age contribute to home value?
2. Is there any correlation between logerror and where a property is located?
3. Can we use latitude and longitude to create meaningful clusters?
4. Is there a relationship between logerror and home size?

## Data Dictionary
| Variable | Meaning |
| -------- | ------- |
| zillow   | Full, original dataframe retrieved from the zillow mySQL database |
| train    | Sample (56%) of zillow used for exploring data and fitting/training models|
| validate | Sample (24%) of zillow used to evaluate multiple models |
| test     | Sample (20%) of zillow used to evaluate the best model |
| logerror | Our target variable; the Zestimate error which we want to minimize |
| tax_value | The property's tax assessed value |
| beds     | Number of bedrooms |
| baths    | Number of bathrooms, including fractional bathrooms |
| fullbaths | Number of full bathrooms |
| latitude | The property's latitude |
| longitude | The property's longitude |
| rooms    | Total number of rooms |
| sq_ft    | Calculated total finished living area |
| fips     | Federal Information Processing Standard code |
| fips_loc | The county name corresponding to the property's FIPS code |
| yearbuilt | The year the property was built |
| age      | The age of the property |
| transactiondate | The date the property was sold |
| cluster  | The cluster the property was assigned |
| X_train  | `train`, but only with clusters and scaled columns to be used for modeling |
| y_train  | `train`, but only the target |
| X_validate | `validate`, but only with clusters and scaled columns to be used for modeling |
| y_validate | `validate`, but only the target |
| X_test   | `test`, but only with clusters and scaled columns to be used for modeling |
| y_test   | `test`, but only the target |


## The Plan

### Wrangle

1. Define a function to acquire the necessary zillow data from the mySQL database.
2. Define a function to clean the acquired zillow data.
3. Define a function to split the cleaned zillow data.
4. Define a function to scale the split zillow data.
5. Define a function to combine the previous steps into a single function.
6. Ensure all functions work properly and add them to wrangle.py file.

### Explore
1. Ask a clear question.
2. Develop null and alternative hypotheses for that question.
3. Use visualizations and statistical tests to find answers.
4. Clearly state the answer to the question and summarize findings.
5. Repeat for a total of at least 4 questions.
6. Explore clusters to potentially use as features for models.
7. Summarize key findings, takeaways, and actions.

### Model
1. Select a metric to use for evaluating models and explain why that metric was chosen.
2. Create and evaluate a baseline model.
    - Find median value of target
    - Set all predictions to that value
    - Evaluate based on selected evaluation metric
3. Develop four models.
4. Evaluate all models on the train sample, note observations.
5. Evaluate the top models on the validate sample, note observations.
6. Evaluate the best performing model on the test sample, note observations.

### Deliver
1. Ensure final report notebook is thoroughly code commented.
2. Ensure notebook contains enough Markdown cells to guide the reader through the report.
3. Write a conclusion summary.
4. Develop actionable recommendations.
5. Suggest next steps for research and/or model improvement.
6. Run final report notebook from beginning to be sure that there are no errors.
7. Submit link to repository containing project files.
8. Submit video of recorded presentation.

## Steps to Reproduce
- In a local repository, create a .gitignore file and add env.py.
- In that same repository, create an env.py file to store the hostname, username, and password that will be used to access the zillow mySQL database.
- Clone this repository, making sure the .gitignore is successfully hiding the env.py file.
- Run all cells in Report.ipynb.
- To see our full, in-depth step-by-step thought process for this project, run all cells in the other Jupyter notebooks in this repository.

---
## Conclusion

### Summary
There are clearly patterns in the data that could help us make better predictions. Our best model was an Ordinary Least Squares model using `tax_value`, `sq_ft`, `age`, and the clusters we made during exploration. The four clusters we used were created using `beds`, `baths`, and `sq_ft`. This model had an RMSE of 0.047 for predicting logerror.

### Recommendations
- Find out what drives home value and Zestimate for homes in middle age bins.
- More complete data collection would be useful here. For example, the `rooms` column has a huge amount of 0 values, but by taking a look at the corresponding bedroom or bathroom values, it is clear there are more than 0 rooms in that property.
- Collect data to get a more representative idea of all locations; there are huge amounts of data for Los Angeles county, but very little from Ventura and Orange which made it difficult to accurately compare across the three counties.

### Next Steps
With more time, we would like to further explore the interactions of location and logerror. There appeared to be some differences in logerror based on which county the property was located in, but our samples were unbalanced in terms of county so we decided to save this branch of exploration for another project.
