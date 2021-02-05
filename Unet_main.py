import tensorflow as tf 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import numpy as np 
import cv2
import os
import glob 
import matplotlib.pyplot as plt
import pandas as pd
import Unet as U
import utils 
import MakeDataset as MD


if __name__ == "__main__":

    # Set the GPU you want to use

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Read the data in 

    train_df = pd.read_csv("/home/smalldan/DeepLearningModels/Git/通用/測試中/data/train.csv")
    valid_df = pd.read_csv("/home/smalldan/DeepLearningModels/Git/通用/測試中/data/valid.csv")

    train_img_paths = train_df[train_df.Y=="train_image"].Path.values
    valid_img_paths = valid_df[valid_df.Y=="valid_image"].Path.values

    train_mask_paths = []
    valid_mask_paths = []

    # Get the corresponding mask image path

    for t in train_img_paths:
        
        mask_path = t.replace("image", "label")
        
        train_mask_paths.append(mask_path)
        
        
    for v in valid_img_paths:
        
        mask_path = v.replace("image", "label")
        
        valid_mask_paths.append(mask_path)
        
    # Make the train/valid dataset

    train_dataset = MD.MakeDataset(train_img_paths, train_mask_paths, shuffle_num=len(train_img_paths) , batch=4)

    valid_dataset = MD.MakeDataset(valid_img_paths, valid_mask_paths, shuffle_num=len(valid_img_paths) , batch=1)

    # Get the Unet model

    model = U.Unet()

    # FIXME: There is a bug in this callback func, ModelCheckpoint. It should save the entire model instead of only the weights.
    #        If you want to get rid of it, you can simply use the "model.save" func.

    # Get the callbacks set

    checkpoint_filepath = '/home/smalldan/DeepLearningModels/Github/Unet'
        
    Mcp = ModelCheckpoint(filepath=checkpoint_filepath, monitor = 'val_accuracy', save_weights_only=False, 
                          verbose = 1, save_best_only = True)

    Csvlogger = CSVLogger('Dan_Unet.csv', separator=',', append=False)

    callbacks = [ Mcp, Csvlogger]

    # Set the hyperparameters

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    epochs = 100

    metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.MeanIoU(num_classes=2)]

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    # Start training

    train_history = model.fit(train_dataset, validation_data = valid_dataset, epochs = epochs, callbacks=callbacks, verbose=1)

    # model.summary()

    # You need to change the save path here
    save_path = "/home/smalldan/DeepLearningModels/Git/通用/測試中/model"

    model.save(save_path)

    # Plot the figures

    os.chdir("/home/smalldan/DeepLearningModels/Git/通用/測試中/")

    utils.plot_model_result(train_history)









