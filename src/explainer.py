import numpy as np
import pandas as pd
from pandas import DataFrame as dataframe
import matplotlib.pyplot as plt
# from more_itertools import windowed
# from saxpy.alphabet import cuts_for_asize
# from saxpy.znorm import znorm
# from saxpy.sax import ts_to_string
# from saxpy.paa import paa
# from saxpy import sax
# from saxpy.sax import sax_via_window
from matplotlib.lines import Line2D
import tqdm

from tensorflow import convert_to_tensor, expand_dims, reduce_max, reduce_sum, float32, GradientTape
from tensorflow import abs as tf_abs
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
# import lime
from lime.lime_tabular import LimeTabularExplainer

from shap import TreeExplainer, DeepExplainer, KernelExplainer, Explainer, GradientExplainer

import tensorflow as tf
class LimeExplainer:

    @staticmethod
    def explain(explainer, model, input_data, background=None, input_labels=None):
        # predict_fn, background, input_data, explainer='LimeTabularExplainer'):

        explainer__ = LimeTabularExplainer(
            training_data=background,
            # feature_names=feature_names,
            # class_names=class_names,
            mode='classification',
            random_state =0,
        )

        predict_fn__ = []
        if hasattr(model, 'predict_proba'):
            predict_fn__ = model.predict_proba
        else: 
            predict_fn__ = model.predict
        
        # Explain a prediction from the model
        scores = LimeExplainer.explain_instances(
            explainer=explainer__, 
            input_data=input_data,
            predict_fn=predict_fn__
        )
        
        scores = LimeExplainer.extract_importances_sorted(scores)

        return explainer__, scores
    
    @staticmethod
    def explain_instances(explainer, input_data, predict_fn):
    
        scores = []
        for instance in tqdm.tqdm(input_data):
            explaination = explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn, # model.predict_proba # Use model.predict for regression
                num_samples=2000,
                num_features=input_data.shape[1]
                )
            scores.append(explaination)
        
        return scores
    

    @staticmethod
    def extract_importances_sorted(scores):
        lime_importances = []
        for importance in scores:
            df = pd.DataFrame(importance.as_map()[1], columns=['idx', 'values']).sort_values(by=['idx'])['values'].values
            lime_importances.append(df)
        return np.asarray(lime_importances)


class ShapExplainer:

    def __init__(self):
        pass
        
    @staticmethod
    def explain(explainer, model, input_data, background=None, input_labels=None):
        if background is not None:
            background = input_data
            
        explainer__ =[]
        if explainer=='TreeExplainer':
            explainer__ = TreeExplainer(model, background)
            scores = explainer__.shap_values(input_data)
            return explainer__, scores[:, :, 1]   ## negative class
        elif explainer == 'Explainer':
            # tf.compat.v1.disable_v2_behavior() #  at the top for TF 2.4+
            explainer__ = Explainer(model, background)
        elif explainer == 'DeepExplainer':
            explainer__ = DeepExplainer(model, background)
        elif explainer == 'GradientExplainer':
            explainer__ = GradientExplainer(model, background)
        elif explainer == 'KernelExplainer':
            explainer__ = KernelExplainer(model, background)
        else:
            print('Explainer not found')

        scores = explainer__.shap_values(input_data)
        return explainer__, scores
            

class SaliencyExplainer:

    @staticmethod
    def explain( explainer, model, input_data, background=None, input_labels=None):
        print(model.summary())
        
        if explainer=='SaliencyMap':
            scores = []
            for idx in zip(tqdm.tqdm(range(input_data.shape[0]))):
                label = None
                if input_labels is not None:
                    label = input_labels[idx]
                # saliency = SaliencyExplainer.explain_instance(
                saliency = SaliencyExplainer.generate_saliency_map_with_true_label(
                    model=model, 
                    input_tensor=input_data[idx], 
                    true_label=label
                )
             
                scores.append(saliency)
            return np.asarray(scores)
        
        else:
            raise Exception(f'{explainer} not found')

        return []
            
        

    # @staticmethod
    # def explain_instance(model, input_data, true_label):
    #     saliency = []
    #     if true_label is None:
    #         saliency = SaliencyExplainer.generate_saliency_map(model, input_data, log=True)
    #         return saliency
    #     saliency = SaliencyExplainer.generate_saliency_map_with_true_label(model, input_data, true_label)
    #     return saliency

    @staticmethod
    def generate_saliency_map_with_true_label(model, input_tensor, true_label):
        """Generates a saliency map for a given input using TensorFlow, focusing on binary cross-entropy loss."""
        input_tensor = convert_to_tensor(input_tensor)
        input_tensor = expand_dims(input_tensor, 0)
        true_label = np.expand_dims(true_label, axis=0)
        with GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = model(input_tensor, training=False)  # Get the model's prediction
            # Calculate binary cross-entropy loss
            loss = binary_crossentropy(true_label, prediction)
    
        # Get the gradients of the loss w.r.t the input image
        gradients = tape.gradient(loss, input_tensor)
        # print('gradients', gradients)
        # Compute the absolute gradients
        gradients_abs = tf.abs(gradients)
        # print('gradients_abs: ', gradients_abs)
        # Sum across the channels to get a single saliency map (if your data is multivariate)
        saliency_map = reduce_sum(gradients_abs, axis=0)
        # print(saliency_map)
        return saliency_map.numpy()  # Assuming a single example; adjust as necessary

    
    @staticmethod
    def plot_saliency_heatmap(input_data, saliency_map, threshold=0.5, title="Saliency Heatmap"):
        """
        Plots the saliency map as a heatmap with normalization and thresholding.
    
        Parameters:
        - saliency_map: A numpy array representing the saliency map to be plotted.
        - threshold: A float representing the threshold value to apply after normalization.
        - title: A string, the title of the plot.
        """
        # input_tensor = input_tensor.numpy()[0]
        
        # Normalize the saliency map
        heatmap = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        
        # Apply threshold
        heatmap = np.array([x if x > threshold else 0 for x in heatmap.flatten()])
        heatmap = np.expand_dims(heatmap, 0)  # Make it 2D for consistent plotting
        # print(heatmap)
        
        # Plotting
        plt.figure(figsize=(14, 2))
        plt.imshow(heatmap, cmap='Reds', aspect='auto', interpolation='nearest', alpha=0.7,
                  extent=[0, input_data.shape[0], input_data.min(), input_data.max()], origin='lower')
        plt.plot(input_data, linewidth=3)
        plt.colorbar(label='Saliency')
        plt.title(title)
        plt.show()


class Explanation:

    @staticmethod
    def make_explanation(explainer, model, input_data, background=None, input_labels=None):
        print(f'Generating {explainer} Explanations...')
        explainer__ = []
        
        if explainer in ['TreeExplainer', 'Explainer', 'DeepExplainer', 'KernelExplainer', 'GradientExplainer']:
            explainer__, scores = ShapExplainer.explain(
                explainer=explainer,
                model=model, 
                input_data=input_data,
                background=background, 
                input_labels=None
            )
        
        elif explainer in  ['LimeTabularExplainer']:
            
            explainer__, scores = LimeExplainer.explain(
                explainer=explainer,
                model=model, 
                input_data=input_data, 
                background=background, 
                input_labels=None
            ) 
        elif explainer in ['SaliencyMap']:
            
            scores = SaliencyExplainer.explain(
                explainer=explainer, 
                model=model, 
                input_data=input_data, 
                background=None, 
                input_labels=input_labels)
        else:
            raise Exception(f'Explainer {explainer} not found')
            return []
        
        return explainer__, scores


    @staticmethod
    def plot_heatmap(input_data, importance_map, threshold=0.5, title="Importance Heatmap", figsize=(14, 2), show=True, path_save = None):
        """
        Plots the saliency map as a heatmap with normalization and thresholding.
    
        Parameters:
        - saliency_map: A numpy array representing the saliency map to be plotted.
        - threshold: A float representing the threshold value to apply after normalization.
        - title: A string, the title of the plot.
        """
        # input_tensor = input_tensor.numpy()[0]
        
        # Normalize the saliency map
        heatmap = (importance_map - np.min(importance_map)) / (np.max(importance_map) - np.min(importance_map))
        
        # Apply threshold
        heatmap = np.array([x if x > threshold else 0 for x in heatmap.flatten()])
        heatmap = np.expand_dims(heatmap, 0)  # Make it 2D for consistent plotting
        # print(heatmap)
        
        # Plotting
        plt.figure(figsize=figsize)
        plt.imshow(heatmap, cmap='Reds', aspect='auto', interpolation='nearest', alpha=0.7,
                  extent=[0, input_data.shape[0], input_data.min(), input_data.max()], origin='lower')
        plt.plot(input_data, linewidth=3)
        plt.colorbar(label='Importance')
        
    
        if path_save:
            # print('save in ', path_save)
            fig.savefig(os.path.join(path_save, f'{title}.pdf'), format='pdf', bbox_inches='tight', dpi=64)
        if show:
            plt.title(title)
            plt.show()
        else: 
            plt.close()


    
    # @staticmethod
    # def generate_saliency_map(model, input_data, log=False):
    #     """Generates a saliency map for a given input using TensorFlow, focusing on the gradients of model predictions."""
    
    #     input_tensor = convert_to_tensor(input_data, dtype=float32)
        
    #     if len(input_tensor.shape) == 1:  # Assuming time series data or flat data with shape [features]
    #         input_tensor = expand_dims(input_tensor, 0)  # Add batch dimension
        
    #     with GradientTape() as tape:
    #         tape.watch(input_tensor)
    #         predictions = model(input_tensor, training=False)  # Get the model's predictions
    #         if log: print('predictions', predictions)
    
    #     # Get the gradients of the predictions w.r.t the input image
    #     gradients = tape.gradient(predictions, input_tensor)
    #     # print (gradients.shape)
    #     # Compute the absolute gradients
    #     gradients_abs = tf_abs(gradients)
    #     if log: print('gradients_abs', gradients_abs)
    #     # Sum across the channels to get a single saliency map (if your data is multivariate)
    #     saliency_map = reduce_max(gradients_abs, axis=0)
    #     if log: print('saliency_map: ', saliency_map)          
        
    #     return saliency_map[0].numpy()  


    # working with true label
    # @staticmethod
    # def generate_saliency_map_with_true_label(model, input_data, true_label, log=False):
    #     """Generates a saliency map for a given input using TensorFlow, focusing on binary cross-entropy loss."""
    
    #     input_tensor = convert_to_tensor(input_data, dtype=float32)
    #     if len(input_tensor.shape) == 1:  # Assuming time series data or flat data with shape [features]
    #         input_tensor = expand_dims(input_tensor, 0)  # Add batch dimension
    #     if len(true_label.shape) == 1:  # Assuming true_label is flat
    #         true_label = expand_dims(true_label, 0)  # Add batch dimension
        
    #     with GradientTape() as tape:
    #         tape.watch(input_tensor)
    #         prediction = model(input_tensor, training=False)  # Get the model's prediction
            
    #         # Calculate binary cross-entropy loss
    #         # loss = tf.keras.losses.binary_crossentropy(true_label, prediction)
    #         loss = categorical_crossentropy(true_label, prediction)
    #         if log: print('prediction: ', prediction, ' loss: ', loss)
    
    #     # Get the gradients of the loss w.r.t the input image
    #     gradients = tape.gradient(loss, input_tensor)
    
    #     # Compute the absolute gradients
    #     gradients_abs = tf_abs(gradients)
        
    #     # Sum across the channels to get a single saliency map (if your data is multivariate)
    #     # saliency_map = tf.reduce_sum(gradients_abs, axis=-1)
    #     saliency_map = reduce_max(tf_abs(gradients), axis=0)
    #     if log: print('saliency_map: ', saliency_map)
    
    #     return saliency_map[0].numpy()  


    # @staticmethod
    # def extract_importance_values(feature_importance_list):
    #     """
    #     Extracts the numerical importance values from a list of feature importances.
    
    #     :param feature_importance_list: A list of tuples where each tuple contains a feature description and its importance as a float.
    #     :return: A list of floats representing the importance values.
    #     """
    #     return np.asarray([importance for _, importance in feature_importance_list], dtype=np.float32)

    # @staticmethod
    # def lime2pandas(instance_map, sort=False):
    #     if sort:
    #         return pd.DataFrame(list(instance_map[1]), columns=['idx', 'scores']).sort_values(by=['idx'])['scores'].values
    #     else:
    #         return pd.DataFrame(list(instance_map[1]), columns=['idx', 'scores'])['scores'].values
