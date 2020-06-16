# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:11:36 2020

Various objects and functions for the hotel cancellations modelling
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from IPython.display import display

### Pre-processing

def fixTarget(df):
    '''
    Preprocessing data before split. Updates the status_reservation in cases where
    the arrival_date == reservation_status_date, indicating guest did not stay.
    If was Check-Out, change to 'No-Show' (assuming cancellations are in advance).

    Adds columns for arrival_date as datetime and converts reservation_status_date
    to datetime as well. We can decide later which date information to use, if any.
    '''

    # Clone the data frame
    df_copy = df.copy()

    # We'll be creating some new columns for arrival date and nights_stay. We can
    # then update reservation_status to more correctly indicate cancellations or
    # no shows.

    # Convert reservation_status_date (day status changed, e.g., checked out) to datetime.
    df_copy['reservation_status_date'] = pd.to_datetime(df_copy['reservation_status_date'],
                                                        infer_datetime_format = True,
                                                        errors='coerce')

    # Combine month, day, and year columns into one, and convert into datetime format.
    df_copy['arrival_date'] = pd.to_datetime(df_copy[['arrival_date_month', 
                                  'arrival_date_day_of_month',
                                  'arrival_date_year']].astype(str).agg('-'.join, axis=1))
    
    # Drop the separate date columns as now redundant
    df_copy = df_copy.drop(columns=['arrival_date_month', 'arrival_date_day_of_month',
                                    'arrival_date_year'])

    # Any time arrival_date == reservation_date, and is currently 'Check-Out, will 
    # change reservation_status to 'No-Show'
    cond = (df_copy['reservation_status_date'] == df_copy['arrival_date']) & \
                                  (df_copy['reservation_status'] == 'Check-Out')
    df_copy.loc[cond, 'reservation_status'] = 'No-Show' 
    
    return df_copy

###### Transformers

class wrangleData(BaseEstimator, TransformerMixin):
    '''
    Clean and wrangle the feature sets

    Parameters:
    remove_dates: drop date information (default = True)
    max_cardinality: drop categorical features above threshold,
                     (defult=None, meaning none are dropped)
    '''

    def __init__(self, remove_dates=True, max_cardinality=None):
        self.remove_dates = remove_dates
        self.max_cardinality = max_cardinality

    def fit(self, X, y=None):
        # There is nothing here to do, as we won't 'learn' anything from the training set
        return self

    def transform(self, X, y=None):
        # Where the action is, processing the raw data set
        X_copy = X.copy()

        # Drop the extra is_canceled feature, as we will use the reservation_status
        # feature as target
        X_copy = X_copy.drop('is_canceled', axis=1)
        
        # Calculate total nights stay and drop week and weekend night counts
        # 0, change them to NaN (missing data)
        X_copy['nights_stay'] = X_copy['stays_in_week_nights'] + X_copy['stays_in_weekend_nights']
        X_copy = X_copy.drop(columns=['stays_in_week_nights', 'stays_in_weekend_nights'])
        X_copy['nights_stay'] = X_copy['nights_stay'].replace(0, np.NaN)
        
        # Add feature whether asssigned_room_type and reserved_room_type differs. (Could be source
        # for cancellations?)
        X_copy['room_type_changed'] = X_copy['assigned_room_type'] != X_copy['reserved_room_type']
        X_copy = X_copy.drop(columns=['assigned_room_type', 'reserved_room_type'])

        # Drop date information (except arrival_date_week_number for seasonality)
        # (Optional)
        if self.remove_dates:
            X_copy = X_copy.drop(columns=['arrival_date', 'reservation_status_date'])

        # Drop categorical features with high cardinality 
        # (Optional)
        if self.max_cardinality:
            cardinality = X_copy.select_dtypes(exclude='number').nunique()
            hc_feat = cardinality[cardinality > self.max_cardinality].index.tolist()
            X_copy = X_copy.drop(columns=hc_feat)
            
        # Drop distribution_channel as overlapping with market_segment
        X_copy = X_copy.drop('distribution_channel', axis=1)
        
        # Drop previous_bookings_not_canceled as overlap with is_repeated_guest and
        # previous_cancellations
        X_copy = X_copy.drop('previous_bookings_not_canceled', axis=1)
        
        # Drop babies (but not hard) and children
        X_copy = X_copy.drop(columns=['babies', 'children'])

        # If ADR is < 50 or > 300, change to NaN
        cond = (X_copy['adr'] > 300) | (X_copy['adr'] < 50)
        X_copy.loc[cond, 'adr'] = np.NaN

        # Adults no > 4
        cond = X_copy['adults'] > 4
        X_copy.loc[cond, 'adults'] = np.NaN

        # Drop agent and company
        X_copy = X_copy.drop(columns=['agent', 'company'])

        # In meal, sc and undefined are the same (no meal plan)
        cond = X_copy['meal'] == 'Undefined'
        X_copy.loc[cond, 'meal'] = 'SC'

        # Have property of the final set of columns available
        self.columns_ = X_copy.columns

        return X_copy

class selectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.features]

    
class excludeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        if len(self.features) > 0:
            X_copy = X_copy.drop(columns=self.features)
        return X_copy

### Metrics

def class_metrics(y_true, y_pred):
    '''
    Generate confusion matrix and precision, recall, and f1 scores, for any
    number of classes.

    Inputs:
    y_true, y_pred: actual and predicted values

    Output:
    Confusion_matrix and scores (as dataframes)
    '''
    # Get the labels
    labels = y_true.unique()

    # And a confusion matrix from sklearn
    cm = confusion_matrix(y_true, y_pred, labels)

    # Get sums on each axis, e.g. TP + FN + FN...
    pred_sums = np.sum(cm, axis=0)
    actual_sums = np.sum(cm, axis=1)
    
    # Calculate metric for each store as dict
    output = {}
    for i in range(cm.shape[0]):
        precision = cm[i][i] / pred_sums[i]
        recall = cm[i][i] / actual_sums[i]
        f1 = 2 * precision * recall / (precision + recall)
        output[labels[i]] = {'precision': precision,
                             'recall': recall, 
                             'f1-score': f1}
    
    # Make outputs into tables
    class_matrix = pd.DataFrame(output).T
    confuse_matrix = pd.DataFrame(cm, columns=labels, index=labels)
    return confuse_matrix, class_matrix


def ROCcurves(y_true, X, model, classes=''):
    '''
    Plot ROC curves for each class 
    Inputs: 
    y_true - the ground truth
    model - estimator (or pipeline)
    classes - from classes_ from the estimator, if in a pipeline. If empty will
              try to get them from the model. Labels need to be in order used by 
              the model.
    '''
    # if classes not specified, try to interrogate model for them
    if len(classes) == 0:
        classes = model.classes_

    # Predict probabilities for each class
    y_h = model.predict_proba(X)

    # Setup the plot
    plt.figure(figsize=(10,7))

    # For each label, make a binomial version and get a ROC curve,
    # and plot it.
    for i, label in enumerate(classes):
        y = y_true == label
        fpr, tpr, thresholds = roc_curve(y, y_h[:,i])
        plt.plot(fpr, tpr, '-o', label=label)
    plt.legend()
    plt.title('ROC Curves')
    plt.xlabel('False positive rates')
    plt.ylabel('True positive rates')
    plt.show()


### Convenient function to test and score model

def tryModel(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f'\n\nTrain score = {train_score * 100} %\nValidate score = {val_score * 100} %\n\n')

    y_hat = model.predict(X_val)
    scores = class_metrics(y_val, y_hat)
    display('Confusion Matrix')
    display(scores[0])
    print('\n\n')
    display('Metrics')
    display(scores[1])

    
def permutationImports(model, X_train, y_train, X_val, y_val):
    # Pipeline for transformers alone
    transformers = make_pipeline(tools.wrangleData(max_cardinality=50),
                                ce.OrdinalEncoder(),
                                SimpleImputer())

    X_train_transformed = transformers.fit_transform(X_train)
    X_train_resample, y_train_resample = RandomUnderSampler(random_state=42).fit_sample(X_train_transformed, y_train)
    X_val_transformed = transformers.transform(X_val)

    # fit the  model
    model.fit(X_train_resample, y_train_resample)

    # We'll look at the importances for both accuracy score and recall
    permuter = PermutationImportance(
        model,
        scoring='accuracy',
        random_state=42
    )

    permuter.fit(X_val_transformed, y_val)

    wrangler = transformers.named_steps['wrangledata']

    print('Accuracy score\n')
    permute_scores = pd.Series(permuter.feature_importances_, wrangler.columns_)
    display(permute_scores.sort_values(ascending=False))
    print('\n')

    plt.figure(figsize=(10, len(wrangler.columns_) / 2))
    permute_scores.sort_values().plot.barh()
    plt.show()

    return permute_scores, permute1_scores    
    
