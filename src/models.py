import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import os

def conv_block(layer, filter, kernel, pool_size, stride, activation='relu', dilation_rate=1, name=''):
    conv = layers.Conv1D(filters=filter, kernel_size=kernel, strides=stride, padding='same', 
                         dilation_rate=dilation_rate, 
                        #  kernel_regularizer='l2', 
                         name=name)(layer)
    if pool_size != 0: conv = layers.MaxPooling1D(pool_size=pool_size)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation(activation)(conv)
    return conv

def resblock(layer, filters, kernel, name='conv'):
    fx = layers.Conv1D(filters, kernel, activation='relu', padding='same', name=name+'_RES1')(layer)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv1D(filters, kernel, padding='same', name=name+'_RES2')(fx)
    out = layers.Add()([layer, fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out


def FCN_Fawaz(input_shape,  num_classes): # Fawaz et. al. (2018) Transfer learning for time series classification # Batch size
    
    # dim_length = input_shape[0]
    
    model_input= layers.Input(input_shape)
    # reshaped_input = layers.Reshape((input_shape, 1))(model_input)
    reshaped_input = layers.Reshape(input_shape + (1,))(model_input)

    conv = conv_block(layer=reshaped_input, filter=32, kernel=3, pool_size=2, stride=1, dilation_rate=1, name='conv_1')
    conv = conv_block(layer=conv, filter=32, kernel=3, pool_size=2, stride=1, dilation_rate=1, name='conv_2')
    conv = conv_block(layer=conv, filter=32, kernel=3, pool_size=2, stride=1, dilation_rate=1, name='conv_3')
    conv = conv_block(layer=conv, filter=32, kernel=3, pool_size=0, stride=1, dilation_rate=1, name='conv_4')
    gap_layer = layers.GlobalAveragePooling1D(name='GAP')(conv)
    model_output = layers.Dense(num_classes, activation='softmax', name='predictions')(gap_layer)
    return tf.keras.models.Model(inputs=model_input, outputs=model_output) 


def conv_network(input_shape,  num_classes):
    input_= layers.Input(input_shape)

    conv = conv_block(layer=input_, filter=64, kernel=3, pool_size=2, stride=1, name='conv_1')
    conv = conv_block(layer=conv, filter=64, kernel=3, pool_size=2, stride=1, name='conv_2')
    conv = conv_block(layer=conv, filter=64, kernel=3, pool_size=2, stride=1, name='conv_3')
    conv = conv_block(layer=conv, filter=64, kernel=3, pool_size=0, stride=1, name='conv_4')
    # conv = conv_block(layer=conv, filter=32, kernel=3, pool_size=0, stride=1, name='conv_5')
    # conv = conv_block(layer=conv, filter=16, kernel=3, pool_size=0, stride=1, name='conv_6')

    gap_layer = layers.GlobalAveragePooling1D(name='GAP')(conv)

    # dense_layer = layers.Dense(64, activation='relu', name='dense_layer_1')(gap_layer)
    # dense_layer = layers.Dense(32, activation='relu', name='dense_layer_2')(dense_layer)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(gap_layer)

    return tf.keras.models.Model(inputs=input_, outputs=out) 

## Baseline network
def simple_cnn(input_shape,  num_classes):
    input_= layers.Input(input_shape)

    conv = layers.Conv1D(filters=64, kernel_size=9, strides=1, dilation_rate=3, padding='same', name='conv_1')(input_)
    # conv = layers.MaxPooling1D(pool_size=2)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='conv_2')(conv)
    # conv = layers.MaxPooling1D(pool_size=2)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_3')(conv)
    # conv = layers.MaxPooling1D(pool_size=2)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)


    conv = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', name='conv_4')(conv)
    # conv = layers.MaxPooling1D(pool_size=0)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    gap_layer = layers.GlobalAveragePooling1D(name='GAP')(conv)

    # dense_layer = layers.Dense(16, activation='relu', name='dense_layer')(gap_layer)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(gap_layer)

    return tf.keras.models.Model(inputs=input_, outputs=out)


def load_callbacks(path, model_name = 'model'):

    loss = tf.keras.callbacks.ModelCheckpoint(os.path.join(path, model_name+'_loss.keras'), 
                                              save_best_only=True, monitor='val_loss', mode='min', verbose=2)
    acc = tf.keras.callbacks.ModelCheckpoint(os.path.join(path, model_name+'_acc.keras'), 
                                                  save_best_only=True, monitor='val_accuracy', mode='max', verbose=2)
    # f1 = tf.keras.callbacks.ModelCheckpoint(os.path.join(path, model_name+'_f1.h5'), 
    #                                               save_best_only=True, monitor='val_f1_score', mode='max', verbose=2)
    # recall = tf.keras.callbacks.ModelCheckpoint(os.path.join(path, model_name+'_recall.h5'), 
    #                                               save_best_only=True, monitor='val_recall', mode='max', verbose=2) 
    # precision = tf.keras.callbacks.ModelCheckpoint(os.path.join(path, model_name+'_precision.h5'), 
    #                                               save_best_only=True, monitor='val_precision', mode='max', verbose=2) 
    # history = os.path.join(path, model_name+'_history.csv')

    history = keras.callbacks.CSVLogger(os.path.join(path, model_name+'_history.csv'))

    return loss, acc, history # loss, acc, f1, recall, precision, history


import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred, average='macro'):
    """
    Calculate the F1 score for multiclass classification with macro and micro options, handling type mismatches.
    
    Parameters:
    y_true (tensor): True labels, one-hot encoded.
    y_pred (tensor): Predictions, either as probabilities or one-hot encoded predictions.
    average (str): 'macro' or 'micro' to specify the averaging method.
    
    Returns:
    float: F1 score.
    """
    # Ensure predictions are one-hot encoded
    y_pred = K.one_hot(K.argmax(y_pred), num_classes=tf.shape(y_pred)[-1])
    # Cast y_true and y_pred to float32 to prevent type mismatch
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # True positives, false positives, and false negatives
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    if average == 'macro':
        # Calculate precision, recall, and F1 for each class
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        # Return average (macro) F1 score across all classes
        return K.mean(f1)
    elif average == 'micro':
        # Calculate totals for precision, recall, and F1 across all classes
        tp_total = K.sum(tp)
        fp_total = K.sum(fp)
        fn_total = K.sum(fn)

        precision_total = tp_total / (tp_total + fp_total + K.epsilon())
        recall_total = tp_total / (tp_total + fn_total + K.epsilon())
        f1_total = 2 * (precision_total * recall_total) / (precision_total + recall_total + K.epsilon())
        return f1_total
    else:
        raise ValueError("average must be either 'macro' or 'micro'")


def plot_CM(y_true, y_pred, title='Confusion Matrix', figsize=(8, 4)):

    cm_test = confusion_matrix(y_true=data_dict['y_test'].astype(np.int32).squeeze(), 
                 y_pred=test_pred_class)
    figure = plt.figure(figsize=figsize)
    sns.heatmap(cm_test, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()
    return cm_test

def make_prediction( data_dict, model_path, class_names, path_save=[], title='Model prediction', figsize=(8,4), log=True):

    ## load model
    best_model = tf.keras.models.load_model(model_path)
    if log : print('Evaluating...')
    scores = best_model.evaluate(data_dict['X_test'], data_dict['y_test_ohe'].toarray())
    if log: print(scores)

    # print("%s: %.2f%%" % (best_model.metrics_names[0], scores[0]))
    # print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

    ## predict test set
    if log : print('Compute predictions: ')
    pred_proba = best_model.predict(data_dict['X_test'])
    pred_class = np.argmax(pred_proba, axis=1)

    # if log: print('classes names: ', class_names)
    if log : print('Calculate confusion matrix: ')
    cm =tf.math.confusion_matrix(labels=data_dict['y_test'].astype(np.int32).squeeze(), 
                 predictions=pred_class, num_classes=len(class_names) ).numpy()
    # cm_test = np.fliplr(cm_test)
    if log : print(cm)

    figure = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    if not log:
        plt.close(figure)
    
    if log :print('Saving results')
    if path_save:
        if log : print('save in ', path_save)
        figure.savefig(os.path.join(path_save, title+'_CM.png'), format='png', bbox_inches='tight')

    if log :print('Generating reports')
    classification_report__ = classification_report(y_pred=pred_class, 
                                                  y_true=data_dict['y_test'].astype(np.int32).squeeze(), 
                                                  labels=class_names, 
                                                  output_dict=True,
                                                  zero_division=0)
    if log :print('Generating information file')
    info={
        'metric_names': list(best_model.metrics_names),
        'scores': scores,
        'pred_proba':pred_proba.tolist(),
        'pred_class': pred_class.tolist(),
        'confusion_matrix': np.asarray(cm).tolist(),
        'classification_report': np.asarray(classification_report__).tolist()
    }
    return info


def plot_grid_ms(ms_dict, path_save= [], name='plot grid', figsize=(12, 20)):
    fig = plt.figure(figsize=figsize)
    for i, idx in zip (range(0, 100), np.random.choice( ms_dict['y_test'].shape[0], 25)):
        # print(i, idx)
        ax = plt.subplot(5, 5, i + 1)
        plt.plot(ms_dict['X_test'][idx])
        plt.title('true label: '+ str(ms_dict['y_test'][idx])+ '\n['+ str(ms_dict['labels_list'][idx])+'] \n prediction ' + str(ms_dict['pred_class'][idx]))

        # plt.ylim([-1.6, 1.6])
        plt.axis("off")
    if path_save:
        fig.savefig(os.path.join(path_save, name+'.png'), format='png', bbox_inches='tight')
    return 

def misclassified_data(data_dict, predictions):
    pred_class = np.argmax(predictions, axis=1)
    y_true = data_dict['y_test'].squeeze().astype(int)
    fp_idx = y_true!=pred_class 
    
    label_list= []
    if 'labels_list' in data_dict.keys():
        label_list = data_dict['test_info']['labels_list'][fp_idx]
    elif 'label_list' in data_dict.keys():
        label_list= data_dict['test_info']['label_list'][fp_idx]
    else: 
        print('Key not found')
        return []
   

    ms_dict = {
        'idx': np.asarray(np.where(fp_idx==True)),
        'X_test': data_dict['X_test'][fp_idx],
        'y_test': data_dict['y_test'][fp_idx],
        'pred_proba': np.asarray(predictions)[fp_idx],
        'labels_list': label_list,
        'pred_class': pred_class[fp_idx]
    }

    return ms_dict

def plot_results(history, type='loss', title='Model', path_save=[], figsize=(15, 3), show=True):
    if type=='loss':
        values = {'loss':history.history.get('loss'), 'val_loss':history.history.get('val_loss')}
    elif type=='acc' or type=='accuracy':
        values = {'accuracy':history.history.get('accuracy'), 'val_accuracy':history.history.get('val_accuracy')}
    elif type=='recall':
        values = {'recall':history.history.get('recall'), 'val_recall':history.history.get('val_recall')}
    elif type=='precision':
        values = {'precision':history.history.get('precision'), 'val_precision':history.history.get('val_precision')}
    elif type=='f1' or type=='f1_score':
        values = {'f1_score':history.history.get('f1_score'), 'val_f1_score':history.history.get('val_f1_score')}
    else:
      print('Invalid type')
      return []

    fig = plt.figure(figsize=figsize)
    plt.plot(pd.DataFrame(values))
    plt.title('Model '+type)
    plt.ylabel(type)
    plt.ylim([0, 1.1])
    plt.xlabel('Epoch')
    plt.legend([type, 'val_'+type], loc='upper left')
    if not show:
        plt.close(fig)

    if path_save:
        # print('save in ', path_save)
        fig.savefig(os.path.join( path_save, title+'_'+type+'.png'), format='png', bbox_inches='tight')
    return 