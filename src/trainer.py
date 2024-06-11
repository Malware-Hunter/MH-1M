from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump, load
import os
import pickle
from time import time
import json

# from tensorflow import device
# import models

def train(X, y, model_name, path_experiment=None):
    model = []
    time_start = time()
    if model_name == 'xgboost':
        model = XGBClassifier()  
    elif model_name == 'svm':
        model = svm.SVC(C=0.5, kernel='poly', cache_size=1024)
        y = np.argmax(y, axis=1)
    elif model_name == 'randomforest':
        model = RandomForestClassifier(n_jobs=-1)
    # elif model_name == 'fcn':
    #     model, time_duration = build_deep_model(
    #         X=X, y=y, num_classes=2, epochs=500, batch_size=16, model_name=model_name, path_experiment=path_experiment)
    #     return model, time_duration
    # elif model_name == 'deepconvlstm':
    #     model, time_duratio = build_deep_model(
    #         X=X, y=y, num_classes=2, epochs=500, batch_size=64, model_name=model_name, path_experiment=path_experiment)
    #     return model, time_duration
    else:
        raise Exception(f'Model {model_name} not found in trainer.train()')

    model.fit(X, y)
    time_end = time()
    time_duration = time_end - time_start

    # report the duration
    print(f'Took {time_duration:.4f} seconds to train the {model_name}')

    return model, time_duration

def eval(X, y_true, model, class_labels, dataset_name, model_name, path_save=None):
    y_pred=[]
    if 'xgboost' in model_name:
        y_pred = model.predict_proba(X)
        y_pred=np.argmax(y_pred, axis=1)
    elif 'svm' in model_name:
        y_pred = model.predict(X)
        # y_true=np.argmax(y_true, axis=1)
    elif 'randomforest' in model_name:
        y_pred = model.predict_proba(X)[1]
        y_pred = np.asarray(y_pred)
        y_pred=np.argmax(y_pred, axis=1)
    elif 'deepconvlstm' in model_name:
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
    elif 'fcn' in model_name:
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        print('Model not found')
        return []
    y_true = np.argmax(y_true, axis=1)
    print(y_true.shape, y_pred.shape)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    plot_confusion_matrix(cm, class_labels=class_labels, path_save=os.path.join(path_save, f'{model_name}-{dataset_name}-confusion_matrix'))
    print()
    report=classification_report(y_true, y_pred, output_dict=True, digits=4)
    print(classification_report(y_true, y_pred, digits=4)) #target_names

    if path_save:
        # fig.savefig(os.path.join(path_save+ '-CM.pdf'), format='pdf', bbox_inches='tight', dpi=64)
        with open(os.path.join(path_save, f'{model_name}-{dataset_name}-report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
    return y_pred, cm, report

def save_model(model, directory, filename):
    """
    Save a scikit-learn model to a directory.

    Parameters:
    - model: The scikit-learn model object to be saved.
    - directory: The directory where the model should be saved.
    - filename: The filename for the saved model (without the file extension).

    Returns:
    - filepath: The full path to the saved model file.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct the full filepath
    filepath = os.path.join(directory, filename + '.joblib')

    # Save the model
    dump(model, filepath)
    
    return filepath

def load_sklearn_model(directory, filename):
    """
    load a scikit-learn model.

    Parameters:
    - directory: The directory where the model should be saved.
    - filename: The filename for the saved model (without the file extension).

    Returns:
    - model: The scikit-learn model object.
    """

    # Construct the full filepath
    filepath = os.path.join(directory, filename + '.joblib')
    
    return load(filepath)


def plot_confusion_matrix(cm, class_labels, path_save=None):
    """
    Plot a confusion matrix using matplotlib and seaborn with raw values, corresponding percentages, and class labels.

    Parameters:
    - cm: The confusion matrix as a 2D array or DataFrame.
    - class_labels: List of labels for the classes corresponding to the axes of the matrix.

    Returns:
    - figure: The matplotlib figure object.
    """
    # Normalize the confusion matrix to get percentages
    cm_normalized = np.asarray(cm).astype('float') / np.asarray(cm).sum(axis=1)[:, np.newaxis]
    
    # Create annotations format combining raw values and percentages
    annotations = np.array([["{0}\n{1:.2%}".format(value, cm_normalized[i, j]) for j, value in enumerate(row)] 
                            for i, row in enumerate(cm)])

    # Create the matplotlib figure
    figure = plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=annotations, cmap=plt.cm.Blues, fmt='', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save+'.pdf', format='pdf', bbox_inches='tight', dpi=64)


# def build_deep_model(X, y, num_classes, epochs, batch_size, model_name, path_experiment):

#     time_start = time()
#     with device('/gpu:0'):

#         if model_name == 'fcn':
#             model_base = models.FCN_Fawaz(
#                 input_shape=(X.shape[1], ), 
#                 num_classes = num_classes, 
#             )
#         elif model_name == 'deepconvlstm':
#             model_base = models.DeepConvLSTM(
#                 input_shape=(X.shape[1], ), 
#                 num_classes = num_classes, 
#                 filters=[64, 64, 64, 64]
#                 )
#         else:
#             raise Exception(f'{model_name} not found in build_deep_model()')
#             return []
        
#         model_base.compile(
#             loss='categorical_crossentropy', 
#             # loss='binary_crossentropy', 
#             optimizer = 'adam', ##tf.keras.optimizers.Adam(),
#             metrics=[
#                 'accuracy', 
#             ]
#         )
        
#         model_base.summary()

#         model_base.fit(
#             X, y,
#             validation_split=0.15,
#             # validation_data=(X_val, y_val),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=models.load_callbacks(
#                 path=path_experiment, 
#                 model_name = model_name),
#         )

#     time_end = time()
#     time_duration = time_end - time_start
#     # report the duration
#     print(f'Took {time_duration:.4f} seconds to train the {model_name}')
#     return model_base, time_duration