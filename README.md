# ML based covid cases nowcasting

[![ML-Pipeline](https://github.com/jakobkolb/ml-covid-nowcasting/actions/workflows/update.yaml/badge.svg?branch=main)](https://github.com/jakobkolb/ml-covid-nowcasting/actions/workflows/update.yaml)

## Description
TLDR: This project trains a logistic regression model on daily vital data (resting heart rate and steps) that is augmented with survey data consisting of symptoms and sex and age as well as covid variant shares to predict covid test results. The trained model is then used to estimate the covid case incidence in the user population over time.

Figures in this README are updated with latest data on a daily basis via Github actions.


## Usage:
from the project root:
* run `poetry install` to setup env and install dependencies,
* run `make notebook` to start jupyter notebook in virtual env.


### Data:
Data consists of daily values for resting heart rate and steps for each user on many (not all) days. In addition, users report approximately once a week on which symptoms they experienced and whether they got tested for COVID during the past seven days. If they took a test, they also report the test result. In addition, users reported age and sex once.

### Feature Construction:
Reported symptoms are coded as follows:
* symptoms: 1 if the user experienced the symptom and 0 if not,
* Age: as age groups in 5 year brackets starting 1935, ending 2010 labeled as integers from 1 to 16, 
* Sex: 1 female, 2 male
* Resting heart rate (rhr): beats per minute once per day
* Steps: number of steps per day once per day

For vital data we construct the following derived features:

We calculate the median over 60 days before the week for which symptoms and test results were reported, if these 60 days contain data on more than 30 days. From that, we subtract the maximum value during the seven days for which test and symptoms were reported (if the week contains data on 3 or more days).
Same for Steps except that instead of the maximum, we subtract the mean during the test week.

### Model:
The Model contains of `sklearn` normal scaler and logistic regression classifiers:

```python
import pandas as pd
from typing import List
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(features: List[str], target: str, data: pd.DataFrame):

    X = features[features].values
    y = features[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)

    return model, X_test, y_test
```

### Model evaluation

To evaluate the models quality, I use the following plots displaying precission and recall vs. decision threshold in A), true positive and false positive rate vs. decision threshold in B, feature importance as regression coefficients in C) and confusion matrix in D).
![Precision, recall, tpr and fpr vs. decision threshold, feature importance and confusion matrix](model/model_metrics.png?raw=true)

To nowcast covid cases I
1) score all data (also the data without test results) with the trained model, 
2) classify the resulting infection probabilities with a threshold of 0.5 
3) exclude positive classifications in weeks after a previous positive classification to only count new cases
4) use a rolling average over 7 days, normalize by all observations on each day and multiply by to calculate the daily incidence per 100.000.

This incidence is compared to reported incidences stratified by vaccination status in the following plot:

![Nowcasted compared to reported 7 day case incidence](model/incidence_nowcast.png?raw=true)
