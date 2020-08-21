import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class Gradient_descent:
        
    def feature_Scalling(self, X_train, y_train):

        '''
        This method is for feature scalling of independent variables and dependent variable.

        Input --->
        X_train: The independent variables as a matrics
        y_train: The dependent variable as a matrics

        Output --->
        return: The feature scalled X_train and y_train matrics

        Author: Sayan Roy
        '''


        # creating the StandarScaler object of sklearn.preprocessing
        sc = StandardScaler()

        # feature scalling of X_train
        X_train = sc.fit_transform(X_train)

        # feature scalling of y_train
        y_train = sc.transform(y_train)

        return X_train, y_train
    
    def feature_Scalling_X(self, X_train):

        '''
        This method is for feature scalling of independent variable.

        Input --->
        X_train: The independent variables as a matrics

        Output --->
        return: The feature scalled X_train and y_train matrics

        Author: Sayan Roy

        '''

        # creating the StandarScaler object of sklearn.preprocessing
        sc = StandardScaler()

        # feature scalling of X_train
        X_train = sc.fit_transform(X_train)
        
        return X_train
    
    def sigmoid(self, z):

        '''
        This method is for sigmoid function (1 / 1 + exp^(-z))

        Input --->
        z = The value that has to be converted

        Output --->
        return: The value between 0 and 1

        Author: Sayan Roy
        '''
        
        # the sigmoid formula
        h = 1 / (1 + np.exp(-z))
        
        return h
    
    def gradient_descent(self, X_train, y_train, epochs = 100, learning_rate = 0.001):


        '''
        This method is for batch gradient descent.

        Input --->
        X_train: The featured scaled independent variables as a matrics
        y_train: The dependent variables as a matrics
        epochs: No. of epochs, default value is 100 (epochs = 100)
        learning_rate = value of the learning rate. By-default value is 0.001 (learning-rate = 0.001)

        Output --->
        Print the loss or error value after every epoch iteration
        return:  total_lost_list, w_list, b
        total_lost_list: The list of error values. It has the values of errors according to the epoch numbers
        w_list: The list of weights according to every epochs. 
        b: The value of bias


        Author: Sayan Roy
        '''


        # initializing the weights as zero
        w = np.zeros(X_train.shape[1])

        # initialiging the bias as zero
        b = 0

        # creating the empty list for error values
        total_lost_list = []

        # creating the empty list for every weights
        w_list = []
        

        # iteration according to the epochs
        for epoch in range(epochs + 1):

            # creating the empty list for storing sigmoid predicted values
            y_hat = []

            # initializing the loss error as zero
            loss = 0

            # creating the empty list for storing the values of difference between y_train and y_hat 
            difference_y_train_y_hat_list = []
            

            # calculating the loss error value
            for i in range(X_train.shape[0]):
                z = b
                
                for j in range(w.shape[0]):
                    z += X_train[i][j] * w[j]
                    
                h = self.sigmoid(z)
                
                y_hat.append(h)
                
                loss += y_train[i] * np.log(y_hat[i]) + (1 - y_train[i]) * np.log(1 - y_hat[i])
                
            total_loss = loss * (-1 / X_train.shape[0])
            
            total_lost_list.append(round(total_loss, 6))
            

            # calculating and then adding the values of difference between  y_train and y_hat in difference_y_train_y_hat_list list
            for i in range(X_train.shape[0]):
                
                difference_y_train_y_hat = (y_train[i] - y_hat[i])
                
                difference_y_train_y_hat_list.append(difference_y_train_y_hat)
                
            # updating the weights 
            for j in range(w.shape[0]):
                d_w = 0
                for k in range(X_train.shape[0]):
                    d_w += difference_y_train_y_hat_list[k] * X_train[k][j]
                    
                w[j] = w[j] + (learning_rate / X_train.shape[0]) * d_w
                
            # updating the bias
            b = b + (learning_rate / X_train.shape[0]) * sum(difference_y_train_y_hat_list)
            

            # adding the updated weights in w_list
            w_list.append(w)

            
            print(f"Epoch({epoch} / {epochs}): Error ---> {round(total_loss, 6)}", "\n\n")
            
            
        return total_lost_list, w_list, b

        