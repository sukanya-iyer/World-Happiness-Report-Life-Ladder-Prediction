# World Happiness ML Prediction
This project predicts "life ladder", or country-level happiness scores, using the World Happiness Report Dataset, implementing the full machine learning cycle: data preprocessing, exploratory data analysis (EDA), model training, and evaluation.
The goal is to identify key factors influencing happiness, enabling governments and organizations to prioritize resources for improving societal wellbeing. Built with Python libraries including Pandas, Scikit-learn, Matplotlib, and Seaborn, the project demonstrates data analytics and regression modeling skills, achieving an R<sup>2</sup> score of 0.80 with a Linear Regression model.

## Problem Overview
- **Objective**: Develop and decide the optimal machine learning model predicting the happiness score for a country ('Life Ladder') using a supervised regression approach
- **Dataset**: World Happiness Report containing features such as: Log GDP per capita, Healthy life expectancy, Freedom to make life choices, and others.
- **Impact**: Insights from the model can guide resource allocation and policy decisions to enhance well-being by focusing on high-impact factors like GDP and social support. 

## Methodology
### Data Preparation and Preprocessing
- **Aggregation**: Grouped data by country, computing mean feature values to simplify analysis.
- **Missing values**: Filled in missing numerical data with column means
- **Handling outliers**: Applied winsorization to cap happiness scores at 1% limits.
- **Scaling** Standardized features using StandardScaler for model compatability
- **Feature selection**: Dropped irrelevant features based on low correlation with happiness label
- **EDA** Visualized happiness distribution using a historgram and correlations using pairpolots and scatterplots to identify key predictors like Log GDP per capita. 

### Modeling 
- #### Models trained:
  - Linear Regression
  - Decision Tree (tuned with GridSearchCV: max_depth=4, min_samples_leaf=25)
  - Random Forest 
  - Gradient Boosting 
- Feature Importance: Log GDP per capita, Healthy life expectancy, and Social support were top predictors across models.

### Evaluation
- #### Metrics: 
  - Root Mean Squared Error (RMSE) and R<sup>2</sup> score on a 25% test set. 
- #### Results:
  - Linear Regression: RMSE = 0.45, R<sup>2</sup> = 0.80 (best performance)
  - Random Forest: RMSE = 0.49, R<sup>2</sup> = 0.77
  - Gradient Boosting: RMSE = 0.49, R<sup>2</sup> = 0.77
  - Decision Tree: RMSE = 0.70, R<sup>2</sup> = 0.52
- #### Analysis:
  - Linear Regression had the best performance of all models due to the dataset's linear relationships. Using only the top features for model training worsened performance, so all relevant features were retained.

## Final results: 
- ### Best Model: Linear Regression (RMSE: 0.45, R<sup>2</sup>: 0.80) 
- ### Key findings:
  - Log GDP per capita (0.82 correlation), Healthy Life Expectancy, and Social Support are most strongly correlated with happiness, guiding resource allocation for societal impact.
- ### Visualizations:
  - Feature importance bar plots and model performance charts highlight key features and model comparisons. 
