import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
import pandas as pd
from pandas import DataFrame as dataframe
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import os
from os.path import join

import shap

def train(x_train, y_train, x_test, y_test, class_names, classifier='xgboost'):
    
    
    model, y_pred, acc, cm, report= [], [], [], [], []
    
    if classifier=='xgboost':
        print('xgboost')
        model = XGBClassifier()
    elif classifier=='randomforest':
        print('randomforest')
        model = RandomForestClassifier(random_state=0)
    # elif classifier=='svm':
    #     print('svm')
    #     model = SVC(gamma='auto')
    # elif classifier=='knn':
    #     model = KNeighborsClassifier(n_neighbors=3)
    # elif classifier=='linear_regression':
    #     model = LinearRegression()
    # elif classifier=='naive_bayes':
    #     model = GaussianNB()
    else:
        print ('Classifier not found')
        return []
    print('training')
    model.fit(x_train, y_train)
    
    print('predict')
    y_pred=model.predict(x_test)
    
    print('evaluation')
    
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    
    # print('confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:", acc)
    
    print('Generating reports')
    report = classification_report(y_true=y_test, y_pred=y_pred, target_names=class_names)
    print(report)

    return model, y_pred, cm, report

def plot_cm(cm, class_names, title='', path_save=[], figsize=(4, 3)):
    fig= plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap='Blues', linewidth=.5, fmt="g") # fmt=".1f")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.xticks(
        ticks=[.5, 1.5], labels = class_names)
    plt.yticks(ticks=[.5, 1.5], labels = class_names)
    plt.tight_layout()
    if path_save:
        plt.savefig(join(path_save, title+'_cm.png'), format='png')
    return

def plot_importance(model, feature_names, threshold=0.01, figsize=(4, 3)):
    fig= plt.figure(figsize=figsize)
    feature_names = np.asarray(feature_names)
    indices = np.argsort(model.feature_importances_)
    importance = dataframe({'features':feature_names[indices], 'values': model.feature_importances_[indices]})
    importance = importance[::-1].reset_index(drop=True)
    sns.barplot(data=importance[importance['values']>threshold],
           y='features', x='values',
           orient= 'h')

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing importance scores")
    plt.yticks(fontsize=6)
    # plt.legend(xg_model.classes_)
    plt.show()
    return importance

def shap_explainer(model, X, y):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return explainer, shap_values

def process_column(column_names, delimiter='.'):
    process_col = [col.strip().replace(';','.').replace(',','.').replace(':','.').replace('->','').replace('/','.').replace(' ', '_').lower() for col in column_names]

    return [col.split(delimiter)[-1] for col in process_col]

def plt_summary_custom(shap_values, X, max_display, feature_names, 
                       figsize=(5,3), path_save=[], title='beeswarm',
                       plot_type = 'bar'):
    #  plot_type = 'bar', 'dot', 'violin'
    
    cmap = LinearSegmentedColormap.from_list("", ["skyblue","indianred"])
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                      max_display=max_display, plot_size=figsize, cmap = cmap,
                      plot_type = plot_type, show=False, 

                     )
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.ylabel('Features' , fontsize=10)
    plt.xlabel('SHAP values impact on model output', fontsize=10)
    if path_save: 
        plt.savefig(join(path_save, title+'_shap_summary_'+plot_type+'.pdf'), format='pdf', dpi=512, bbox_inches='tight')
    plt.show()

    return 

def plt_beeswarm_custom(shap_values, max_display, figsize=(5,3), path_save=[], title='beeswarm'):
    cmap = LinearSegmentedColormap.from_list("", ["skyblue","indianred"])

    shap.plots.beeswarm(
        shap_values, max_display=30, 
        plot_size=[5,3], show=False, 
        color= cmap#plt.get_cmap("magma")
    )
    # fig, ax = plt.gcf(), plt.gca()
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.ylabel('Features' , fontsize=10)
    plt.xlabel('SHAP values impact on model output', fontsize=10)
    if path_save: 
        plt.savefig(join(path_save, title+'_shap_beeswarm.pdf'), format='pdf', dpi=512, bbox_inches='tight')
    plt.show()

    return

def get_feature_importances(shap_values, column_names):
    feature_names = list(column_names)
    vals = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature','importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
    feature_importance
    return feature_importance.reset_index()