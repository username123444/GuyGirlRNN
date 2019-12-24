import random
import numpy as np
import matplotlib.pyplot as plotter
import os
import sys
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import shutil
import pandas
import math

class Trainer:
    BATCH_SIZE_DEFAULT = "dataset_size"
    def __init__(self, training_dir, categories, model_name="trained.model", image_size=100, log_dir="logs", clear_output=True):
        self.__data_dir = training_dir
        self.__categories = categories
        self.__image_size = image_size
        self.__training_data = []
        self.__log_dir = log_dir
        self.__model_name = model_name
        self.__current_train_count = 0
        if(clear_output is True):
            if(os.path.exists("labels.pickle")):
                os.remove("labels.pickle")
            if(os.path.exists("feature_sets.pickle")):
                os.remove("feature_sets.pickle")
            if(os.path.exists(self.__model_name)):
                shutil.rmtree(self.__model_name)
            if(os.path.exists("logs")):
                shutil.rmtree("logs")

    
    def __create_image_training_data(self):
        for category in self.__categories:
            #Get the path of the category
            path = os.path.join(self.__data_dir, category)
            #Get the index of the current category
            class_num = self.__categories.index(category)
            #Get all of the images in the given path
            for image in os.listdir(path):
                #Create an image array by reading its content
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                #Resize the image in a new array
                new_array = cv2.resize(image_array, (self.__image_size, self.__image_size))
                #Append our training data
                self.__training_data.append([new_array, class_num])
            #Return the training data
        return self.__training_data
        
    def create_new_image_data(self):
        #Create the training data
        self.__create_image_training_data()
        #Shuffle the training data
        random.shuffle(self.__training_data)
        #Create the feature set
        feature_sets = []
        #Create our labels
        labels = []
        #Loop through the feature sets and labels in the training data
        for feature, label in self.__training_data:
            #Update the features
            feature_sets.append(feature)
            #Update the labels
            labels.append(label)
        #Make sure our feature sets is a numpy array
        feature_sets = np.array(feature_sets).reshape(-1, self.__image_size, self.__image_size, 1)
        #Also make sure our labels are a numpy array
        labels = np.array(labels)
        #Open our feature sets save
        pickle_feature_sets_out = open("feature_sets.pickle", "wb")
        #Save the feature sets
        pickle.dump(feature_sets, pickle_feature_sets_out)
        #Close the feature sets save
        pickle_feature_sets_out.close()
        #Open our labels save
        pickle_labels_out = open("labels.pickle", "wb")
        #Save our labels
        pickle.dump(labels, pickle_labels_out)
        #Close the labels save
        pickle_labels_out.close()
    
    def train_image_model(self, train_times=1, validation_split=0.3, epochs=100, auto_save_trained=False, batch_size=BATCH_SIZE_DEFAULT, create_data=True, dense_layers = [0], layer_sizes = [64], convolusion_layers = [3]):
        #Check if batch size is dataset_size
        if(batch_size is "dataset_size"):
            #Initiate a file counter
            __file_counter = 0
            #Walk through all of the files in the data dir
            for root, dirs, files in os.walk(self.__data_dir):
                #Loop through the files
                for file in files:
                    #Update our file counter
                    __file_counter += 1
            #Set the batch size to the file counter
            batch_size = __file_counter
        #Check if the developer wants to create the feature sets and labels
        if(create_data):
            #Create the training data
            self.create_new_image_data()
            
        #Create an old log dir variable
        __old_log_dir = self.__log_dir
        #Load the feature sets
        feature_sets = pickle.load(open("feature_sets.pickle", "rb"))
        #Divide the feature sets by 255
        feature_sets = feature_sets / 255.0
        #Load the labels
        labels = pickle.load(open("labels.pickle", "rb"))
        #Create a total dense, total layer sizes, and total convolusion layers
        total_dense, total_layer, total_convolusion = (0, 0, 0)
        for d in dense_layers:
            total_dense += 1
            for l in layer_sizes:
                total_layer += 1
                for c in convolusion_layers:
                    total_convolusion += 1
        current_dense, current_layer, current_convolusion = (0, 0, 0)
        #Loop through the dense layers list
        for dense_layer in dense_layers:
            current_dense += 1
            #Loop through the layer sizes
            for layer_size in layer_sizes:
                current_layer += 1
                #Loop through our convolusion layers
                for convolusion_layer in convolusion_layers:
                    current_convolusion += 1
                    #Create a name for logging our trained accuracy, and other data
                    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(convolusion_layer, layer_size, dense_layer, int(time.time()))
                    #Update the log directory
                    self.__log_dir = self.__log_dir + "/{}".format(NAME)
                    #Create the log directory
                    os.makedirs(self.__log_dir)
                    #Create our tensorboard callback
                    tensorboard = TensorBoard(log_dir=self.__log_dir, profile_batch = 100000000)
                    #Create our model
                    model = Sequential()

                    #Create the input layer
                    model.add(Conv2D(layer_size, (3, 3), input_shape=feature_sets.shape[1:]))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2,2)))

                    #Loop through the current range of convolusion layers
                    for l in range(convolusion_layer - 1):
                        #Create a hidden layer
                        model.add(Conv2D(layer_size, (3, 3)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2,2)))
                    
                    #Flatten our network
                    model.add(Flatten())

                    #Loop through our dense layer range
                    for l in range(dense_layer):
                        #Add the final hidden layer
                        model.add(Dense(layer_size))
                        model.add(Activation("relu"))
                    #Create our output layer
                    model.add(Dense(1))
                    model.add(Activation("sigmoid"))

                    #Compile the model
                    model.compile(loss="binary_crossentropy",
                                optimizer="adam",
                                metrics=["acc"])
                    #Train the neural network
                    model.fit(feature_sets, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[tensorboard])
                    #Save the model
                    model.save(self.__model_name)
                    #Revert to old log dir
                    self.__log_dir = __old_log_dir
                    #Open our feature sets save
                    pickle_feature_sets_out = open("feature_sets.pickle", "wb")
                    #Save the feature sets
                    pickle.dump(feature_sets, pickle_feature_sets_out)
                    #Close the feature sets save
                    pickle_feature_sets_out.close()
                    #Open our labels save
                    pickle_labels_out = open("labels.pickle", "wb")
                    #Save our labels
                    pickle.dump(labels, pickle_labels_out)
                    #Close the labels save
                    pickle_labels_out.close()
                    print("CURRENT DENSE:  %s out of %s"%(str(current_dense), str(total_dense)))
                    print("CURRENT LAYER SIZES: %s out of %s"%(str(current_layer), str(total_layer)))
                    print("CURRENT CONVOLUSION LAYERS: %s out of %s"%(str(current_convolusion), str(total_convolusion)))
                    self.__current_train_count += 1
        if(self.__current_train_count < train_times):
            self.train_image_model(train_times=train_times, validation_split=validation_split, epochs=epochs, auto_save_trained=auto_save_trained, batch_size=batch_size, create_data=create_data, dense_layers=dense_layers, layer_sizes=layer_sizes, convolusion_layers=convolusion_layers)


    def test_image(self, image_path, move_after_testing=True, train_after_testing=True):
        #Load the image array
        image_array = self.prepare_image(image_path)
        #Divide the image array by 255
        image_array = image_array / 255.0
        #Load the model
        model = tf.keras.models.load_model(self.__model_name)
        #Make a prediction
        prediction = model.predict([image_array])
        #Round the prediction
        rounded_prediction = int(np.round(prediction))
        #Check if we should move after testing
        if(move_after_testing is True):
            #Create a new path variable with the new path being in our dataset folder
            __new_path = "{}/{}/{}".format(self.__data_dir, self.__categories[rounded_prediction], os.path.basename(image_path))
            #Move the file
            shutil.copy(image_path, __new_path)
        #Check if we should train after testing
        if(train_after_testing is True):
            #Train again
            print(self.__categories[prediction])
            self.fit_image_to_saved_model(__new_path, prediction)
        #Return the prediction and the category it belongs to
        return [prediction, self.__categories[rounded_prediction]]
    
    def prepare_image(self, image_path):
        #We need to resize and grayscale our image
        #Create the image array from the content of the image
        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #Create a new array for resizing
        new_array = cv2.resize(image_array, (self.__image_size, self.__image_size))
        #Return the reshaped new array
        return new_array.reshape(-1, self.__image_size, self.__image_size, 1)
    
    def fit_image_to_saved_model(self, image_path, class_number,train_times=1, validation_split=0.3, auto_save_trained=False, batch_size=10, create_data=True, dense_layers = [0], layer_sizes = [64], convolusion_layers = [3]):
        #Ensure that the image cannot be fit if the feature sets are missing
        if(os.path.exists("feature_sets.pickle") is False):
            raise Exception("feature_sets.pickle is missing")
        #Ensure that the image cannot be fit if the labels are missing
        if(os.path.exists("labels.pickle") is False):
            raise Exception("labels.pickle is missing")
        #Get the content of the image via cv2
        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #Divide the image array by 255
        image_array = image_array / 255.0
        #Resize the image
        new_array = cv2.resize(image_array, (self.__image_size, self.__image_size))
        self.__training_data.clear()
        #Update our training data
        self.__training_data.append([new_array, class_number])
        #Save the data
        #Shuffle the training data
        random.shuffle(self.__training_data)
        #Create the feature set
        feature_sets = pickle.load(open("feature_sets.pickle", "rb"))
        #Create the labels
        labels = pickle.load(open("labels.pickle", "rb"))
        #Open our feature sets save
        pickle_feature_sets_out = open("feature_sets.pickle", "wb")
        #Save the feature sets
        pickle.dump(feature_sets, pickle_feature_sets_out)
        #Close the feature sets save
        pickle_feature_sets_out.close()
        #Open our labels save
        pickle_labels_out = open("labels.pickle", "wb")
        #Save our labels
        pickle.dump(labels, pickle_labels_out)
        #Close the labels save
        pickle_labels_out.close()
        #Get the feature sets again
        feature_sets = pickle.load(open("feature_sets.pickle", "rb"))
        #Divide the feature sets by 255
        feature_sets = feature_sets / 255.0
        #Get the labels again
        labels = pickle.load(open("labels.pickle", "rb"))
        #Train the network
        if(os.path.exists(self.__model_name)):
            __old = self.__log_dir
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(0, 1, 2, int(time.time()))
            #Update the log directory
            self.__log_dir = self.__log_dir + "/{}".format(NAME)
            os.makedirs(self.__log_dir)
            #Get the model
            model = tf.keras.models.load_model(self.__model_name)
            #Compile the model
            model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=["acc"])
            tensorboard = TensorBoard(log_dir=self.__log_dir, profile_batch = 100000000)
            #Train the neural network
            model.fit(feature_sets, labels, batch_size=batch_size, epochs=2, validation_split=validation_split, callbacks=[tensorboard])
            #Save the model
            model.save(self.__model_name)
            self.__log_dir = __old
    
    def get_categories(self):
        #Return the categories
        return self.__categories
    
    def get_training_data(self):
        #Return the training data
        return self.__training_data
    