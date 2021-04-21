#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:06:07 2020

@author: sa1
"""
import random
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from __main__ import dataset_ID,batch_size,iid_or_not #Import the dataID used and if it is iid or not
import gc
import tensorflow_addons as tfa
import sys

#%%                                                                     CLASSES
#%% CLIENT CLASS
class Clients(object): 
    def __init__(self,loss_obj,x,y):
        self.x=x
        self.y=y
        self.og_x=None #og_x and og_y are used to keep permanent unaltered images
        self.og_y=None
        self.shuffle_images_counter=0 #Counts how many times data has been shuffled
        self.x_test=None
        self.y_test=None
        self.client_ID=0 #Client ID
        self.loss_obj=loss_obj
        self.one_hot=False
        self.numbers_present=[]
        self.global_epoch=0
        self.batch_size=batch_size  #Indicative of the number of batches the data should be split into
        self.batch_samples=None      #How many samples per batch 
        self.allow_shuffle=True #Flag for shuffling data
        self.num_classes=10
        self.batch_index=0#To indicate which batch number should be fed into mini batch GD algorithm
        self.local_epoch=-1#Counts how many epoch passes have occered locally
        self.data_subset_size=1 #To control what % of the total data should be in the client
        self.epoch_tracker=False #Is true when 1 local epoch is done. When all clients have this true, One global epoch is finished
        self.which_batch=0
        self.batches_of_x=0 #List of data sliced into batches
        self.batches_of_y=0 #List of data sliced into batches
        self.imrow=28
        self.imcol=28
        self.num_channels=1
        self.use_train_or_test=1 #1 train, 0 test, 2 og_data
        self.weights=None
        self.use_mini_batches=True #Uses minibatches while computing loss, instead of the entire data set
        self.Loss=[]
        self.Grads=[]

    @property
    def shape(self):
        return self.get_shape_info()
#  GET SHAPE INFORMATION
#This function returns the shape of the neural network in each client
    def get_shape_info(obj):
        w=obj.model.get_weights()
        how_many_layers=np.shape(w)
        model_shape=[]
        for i in range(how_many_layers[0]):
            temp=np.shape(w[i])
            model_shape.append(temp)
        return model_shape

    @property
    def compile_neural_network(self):
        # Creating a Sequential Model and adding the layers
        if dataset_ID=='0':
            self.model = tf.keras.Sequential([
              tf.keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=(self.imrow,self.imcol,self.num_channels),kernel_initializer='glorot_uniform'),
              tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128,kernel_initializer='glorot_uniform'),
              tf.keras.layers.LeakyReLU(alpha=0.1),
              # tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(self.num_classes,kernel_initializer='glorot_uniform'),
              tf.keras.layers.LeakyReLU(alpha=0.1),
              tf.keras.layers.Softmax(axis=-1)
            ])
        elif dataset_ID=='1':
            self.model = tf.keras.Sequential([
              tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),kernel_initializer='glorot_uniform'), # fashion mnist
              tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128,kernel_initializer='glorot_uniform'),
              tf.keras.layers.LeakyReLU(alpha=0.1),
              # tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(self.num_classes,kernel_initializer='glorot_uniform'),
              tf.keras.layers.LeakyReLU(alpha=0.1),
              tf.keras.layers.Softmax(axis=-1)
#              tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),kernel_initializer='glorot_uniform'),
#              tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),kernel_initializer='glorot_uniform'), # fashion mnist
#              tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#              tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='glorot_uniform'),
#              tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#              tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='glorot_uniform'),
#              tf.keras.layers.Flatten(),
#              tf.keras.layers.Dense(64, activation='relu',kernel_initializer='glorot_uniform'),
#              # tf.keras.layers.LeakyReLU(alpha=0.1),
#              # tf.keras.layers.Dropout(0.2),
#              tf.keras.layers.Dense(self.num_classes,activation='relu',kernel_initializer='glorot_uniform'),
#              # tf.keras.layers.LeakyReLU(alpha=0.1),
#              tf.keras.layers.Softmax(axis=-1)
            ])
        # self.model.compile(optimizer='adam',loss=loss_obj, metrics=['sparse_categorical_accuracy'])
        sgd_optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        self.model.compile(optimizer=sgd_optimizer,loss=self.loss_obj, metrics=['accuracy'])
        # self.model.compile(optimizer='SGD',loss=loss_obj, metrics=['accuracy'])
        

    @property
    #Holds how many samples are present in client
    def num_samples(self):
        return self.y.shape[0]
    
    @property
    def vector_weights(self):
        [temp, _]=Vectorize_W(self)
        return temp
    
    @property
    def parameters_per_layer(self):
        [_,temp]=Vectorize_W(self)
        return temp
    @property
    def training_samples_per_class(self):
        if type(self.y.numpy())==type(None):
            return None
        index_of_classes=np.argmax(self.y.numpy(),axis=1) if self.one_hot else self.y.numpy() #Convert onehot to integer
        return [np.sum(index_of_classes==j)for j in self.numbers_present]
    @property
    def testing_samples_per_class(self):
        index_of_classes=np.argmax(self.y_test.numpy(),axis=1) if self.one_hot else self.y.numpy() #Convert onehot to integer
        return [np.sum(index_of_classes==j)for j in self.numbers_present]
    
    @property
    def samples_per_class(self):
        index_of_classes=np.argmax(self.og_y.numpy(),axis=1) if self.one_hot else self.og_y.numpy() #Convert onehot to integer
        return [np.sum(index_of_classes==j)for j in self.numbers_present] #index_of_classes==j is a boolean vector. True+True+False+True=3 in python
    @property
    def x_batch(self):
        if self.batch_index>=self.batch_size-1:
            """if the batch index is above batch size or if it is the first iteration"""
            if self.batch_index>self.batch_size-1:
                self.batch_index=0 #Reset batch count
                self.local_epoch+=1
            self.epoch_tracker=True #At least one local epoch is done.
        if (self.local_epoch==-1) or (self.allow_shuffle):
            # # print("\n Local epoch", self.local_epoch, " done\n")
            if self.allow_shuffle: #Server dictates shuffling data after 1 global epoch
                # print('Shuffling Data in client',self.client_ID)
                Clients.Shuffle_Tensor_Data(self) #Shuffle data in clients
                self.allow_shuffle=False
                self.epoch_tracker=False
            self.local_epoch=0#Reset Global Epochs
            Clients.slice_to_batches(self) #Create/Recreate batches
        # print("Batch {} Client {} ".format(self.batch_index,self.client_ID))
          
        x_slice=self.batches_of_x[(self.which_batch)%self.batch_size] # """Changed here """
        # print("Gradient called on batch ",self.which_batch)
        return x_slice
    @property
    def y_batch(self):
        y_slice=self.batches_of_y[(self.which_batch)%self.batch_size] 

        return y_slice
    @property
    def loss(self):
        if self.use_mini_batches==True:
            x=self.x_batch
            y=self.y_batch
        else:
            if self.use_train_or_test==1:
                x=self.x
                y=self.y
            elif self.use_train_or_test==0:
                x=self.x_test
                y=self.y_test
            else:
                x=self.og_x
                y=self.og_y
        predictions=self.model(x, training=False)
        computed_loss=self.loss_obj(y_true=y, y_pred=predictions)
        return computed_loss
    
    def slice_to_batches(self):
        """This function slices the data in self.x and self.y 
        and stores it in self.batches_of_x and self.batches_of_y """
        extra_data=self.x.numpy().shape[0]%self.batch_size
        if extra_data !=0:
            #Discard any excess data so that the data can be split into even number of batches
            reduce_to=self.x.numpy().shape[0]-extra_data 
        else:
            reduce_to=self.x.numpy().shape[0]
            
        imrow=self.x.numpy().shape[1]
        imcol=self.x.numpy().shape[2]
        rgb_chan=self.x.numpy().shape[3]
        how_many_classes=self.y.numpy().shape[1]
        #Slice x
        new_x=tf.slice(self.x,[0,0,0,0],[reduce_to,imrow,imcol,rgb_chan]) #Slice out extra data so it can be split into batches.
        self.batch_samples=reduce_to
        #Slice y
        new_y=tf.slice(self.y,[0,0],[reduce_to,how_many_classes]) #Slice out extra data so it can be split into batches.
        self.batches_of_y=tf.split(new_y,self.batch_size,axis=0)
        self.batches_of_x=tf.split(new_x,self.batch_size,axis=0)
        
    def Shuffle_Tensor_Data(self):
        #Create a tensor sequence to act as indices
#Shuffle train data
        data_index=tf.range(start=0,limit=self.x.shape[0],dtype=tf.int32) 
        #Get the new shuffled indices
        shuffled_index=tf.random.shuffle(data_index)
        #Reconstruct x, and y in the new shuffled index way
        self.x=tf.gather(self.x,shuffled_index,axis=0) 
        self.y=tf.gather(self.y,shuffled_index,axis=0)
#Shuffle OG Data
        data_index=tf.range(start=0,limit=self.og_x.shape[0],dtype=tf.int32) 
        #Get the new shuffled indices
        shuffled_index=tf.random.shuffle(data_index)
        #Reconstruct x, and y in the new shuffled index way
        self.og_x=tf.gather(self.og_x,shuffled_index,axis=0) 
        self.og_y=tf.gather(self.og_y,shuffled_index,axis=0)
        
        self.shuffle_images_counter+=1
#%%         SERVER CLASS    
class Server(object):
    def __init__(self, cl):
        self.name=None
        self.lamb_vector=None
        self.optimiser=None
        self.samples_in_clients=None
        self.grad_W=None #Empty list to hold gradient info for each client wrt weights
        self.Losses=None #Loss:=  client x loss w.r.t each wt
        self.cl=cl #Client objects
        self.Weights=None #Query weight matrix to be sent to clients        
        self.client_index_minimization=None  #The index of the client whose alpha is being minimised
        self.gamma=0 #Regularizer weight
        self.single_bit=False
        self.va=0 #Cumulative sum of past gradients for alphas (used for rmsprop, adam, adgrad, etc)
        self.vw=0 #Cumulative sum of past gradients for weights (used for rmsprop, adam, adgrad, etc)
        self.D_a=0 #Exponential moving average of squared weight differences (adadelta)
        self.D_w=0 #Exponential moving average of squared weight differences (adadelta)
        self.W_olda=0 #Previous Weight Value Alpha
        self.W_oldw=0 #Previous Weight Value Weights
        self.alph_epsilon=0.1
        self.Alpha_matrix = None
    @property
    def number_of_clients(self):
        return np.shape(self.cl)[0]

    def client_weights(self,j):
        #Returns the vectorized weights of a particular client
        if j >self.number_of_clients-1:
            raise ValueError("Client index out of bounds")
        [wts,_]=Vectorize_W(self.cl[j])
        return wts
    @property 
    def shape(self):
        return self.get_shape_info()
    @property
    def get_parameters(self):
        """Function returns the needed parameters of the server """
        parameters=[self.number_of_clients,self.Alpha_matrix,
                     self.lamb_vector]
        return parameters   
    @property
    def summary(self):
        print("Number of client objects\n ",self.number_of_clients)
        print("Alpha Matrix \n",self.Alpha_matrix)
#        print("Discrepancy Matrix \n", self.disc_matrix)
        print("Lambda Vector \n",self.lamb_vector)
        print("Delta Vector \n",self.delta)


    def client_loss(self,j):
        #Returns the loss vector of client j with respect to all the other client's weights
        if j >self.number_of_clients-1:
            raise ValueError("Client index out of bounds")
        if self.Losses==None:
            raise ValueError("Loss Matrix is empty")
        return self.Losses[j]

    def client_grad(self,j):
        #Returns the gradient of of client j with respect to all the other client's weights
        if j >self.number_of_clients-1:
            raise ValueError("Client index out of bounds")
        if self.grad_W==None:
            raise ValueError("Gradient Matrix is empty")
        return self.grad_W[j]

    def alpha(self,i,j):
        #returns the alpha value between client i and j
        if i or j >self.number_of_clients-1:
            raise ValueError("Client index out of bounds")
        if self.Alpha_matrix.all()==None:
            raise ValueError("Alpha Matrix is empty")
        return self.Alpha_matrix[i][j]

    def finalise_client_weights(self,W):
        for i in range(self.number_of_clients):
            w=W[i]
            w=De_Vectorize_W(w,self.cl[i])
            self.cl[i].model.set_weights(w)

#%%                                                                     FUNCTIONS
#%% VECTORIZE WEIGHTS
"""This function converts the weights of a model's neural network into a vector,
depending on the shape of the object's neural net"""
def Vectorize_W(obj):
#Count the number of trainable parameters in the nerual net

    parameters_in_layer=1
    wts_biases_in_each_layer=np.zeros((len(obj.shape),1))
    for i in range(0,len(obj.shape)-1,2):#Picking only the weights, not the biases
        wts=obj.shape[i] #The shape of weights in the i'th layer
        bias=obj.shape[i+1]#The shape of biases in the i'th layer
        for j in wts:
            parameters_in_layer=parameters_in_layer*j
        
        wts_biases_in_each_layer[i]=parameters_in_layer
        wts_biases_in_each_layer[i+1]=bias[0]
        parameters_in_layer=parameters_in_layer+bias[0] #nmbr_of_wts+number_of_biases

        parameters_in_layer=1 #Reset for the next layer
    #End of for loops
        
#Filling up the vector
    wts_biases_in_each_layer=wts_biases_in_each_layer.astype(int) #convert to integer array
    W=np.zeros((np.sum(wts_biases_in_each_layer,axis=None),1))
    itr=0
    for i in range(0,len(obj.shape)-1,2):
        wts=obj.model.get_weights()[i]   #The weights in the i'th layer
        bias=obj.model.get_weights()[i+1]#The biases in the i'th layer
        
        nmbr_of_wts=wts_biases_in_each_layer[i]
        nmbr_of_biases=wts_biases_in_each_layer[i+1]
        
        W[itr:itr+nmbr_of_wts[0]]=wts.reshape((wts_biases_in_each_layer[i][0],1))
        itr=itr+nmbr_of_wts[0]
        W[itr:itr+nmbr_of_biases[0]]=bias.reshape(wts_biases_in_each_layer[i+1][0],1)
        itr=itr+nmbr_of_biases[0]
    return W, wts_biases_in_each_layer
#%%  DE-VECTORIZE
#Reshaping a vector of weights and biases to fit in the neural network model
def De_Vectorize_W(W,obj):
    if type(W) is list: #For a single client case, this might be passed as a list
        W=W[0]
    w=[]
    number_of_weights=obj.parameters_per_layer
    itr=0
    j=0
    for i in number_of_weights:
        i=i[0] #Converting i to an integer from int array of 1 number
        temp=W[itr:(itr+i)]
        temp=temp.reshape((obj.shape[j]))
        w.append(temp)
        j=j+1
        itr=itr+i
    return w

#%% GETTING DATA
def Get_MNIST_Data(Categorical):
    """ This function returns the MNIST dataset as numpy arrays, split into 
    60,000 training samples and 20,000 testing samples"""
    if dataset_ID=='0':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_ID=='1':
#        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    data_info=np.shape(x_train)
    number_of_train_images=data_info[0]
    length=data_info[1]
    bred=data_info[2]
    num_channels=1
    if len(data_info)==4:
        num_channels=data_info[3]
    number_of_test_images=np.shape(x_test)[0]
    x_train = x_train.reshape(number_of_train_images, length, bred, num_channels)
    x_test = x_test.reshape(number_of_test_images, length, bred, num_channels)
    input_shape = (length,bred, num_channels)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    
    if Categorical:
        y_train=to_categorical(y_train, num_classes=None) #Converting to one hot
        y_test=to_categorical(y_test, num_classes=None) #Converting to one hot
    
    return x_train, y_train,x_test, y_test,input_shape
#%% TAKE MNIST DATA
def Take_MNIST_Data(numbers_to_be_sent,percentage,one_hot,**kwargs):
    """from_MNIST_Database: A boolean. If true, it extracts data from the MNIST dataset
                            If false, it takes data from x and y
       x,y: The dataset from which classes must be extracted"""
    #False indicates no one-hot encoding of y. It will get encoded later in the code
    from_MNIST_Database=kwargs.get("from_MNIST_Database",True)
    input_shape=None
    if from_MNIST_Database:
        [x_train, y_train,x_test, y_test,input_shape]=Get_MNIST_Data(False)
        no_test_data=False
    else:
        no_test_data=True
        x_train=kwargs.get("x",-1)
        y_train=kwargs.get("y",-1)
        if type(y_train)=='int':
            raise ValueError("Enter tensors x and y")
    x=np.empty(np.shape(x_train), dtype = np.float32)
    y=np.empty(np.shape(y_train), dtype = np.float32)
    how_many=np.empty(len(y), dtype = np.float32)
    fill_till=0
    iteration=0
    x1=[]
    y1=[]
    for i in numbers_to_be_sent:
        tempx=x_train[np.squeeze(y_train==i)]
        if not no_test_data: #If there a test dataset
            temptestx=x_test[np.squeeze(y_test==i)] 
            tempx=np.concatenate((tempx, temptestx),axis=0)
        x1.append(tempx)
        
        tempy=y_train[np.squeeze(y_train==i)]
        if not no_test_data:        
            temptesty=y_test[np.squeeze(y_test==i)]
            tempy=np.concatenate((tempy, temptesty),axis=0)
        y1.append(tempy)        

        fill_till=fill_till+len(tempy)
        how_many[iteration]=len(tempy)
        iteration=iteration+1
        
    x2=np.vstack(x1)
    y2=np.concatenate(y1)
    x=x[:fill_till]
    y=y[:fill_till]
    #SHUFFLING THE DATA
    how_many_to_pick=int(np.floor(percentage*len(y2)))
    [x,y]=My_Shuffle(x2,y2,how_many_to_pick)
    #CONVERTING TO TENSORS
    if one_hot:
        y=to_categorical(y, num_classes=10) #Converting to one hot
    x=tf.constant(x, dtype=tf.float32)
    y=tf.constant(y, dtype=tf.float32)
    return x,y,input_shape
#%% INITIALISE CLIENT DATA    
def Initialise_Client_Data(cl,numbers_to_be_sent,percentage,one_hot):
    """ 
    Usage=> 
    Initialise_Client_Data(client_object,[list of numbers to assign],Percentage of numbers from trainingdata to pick)
    eg.) Initialise_Client_Data(cl1,[1,2],0.43)
    will initialise client 1 with images of 1 and 2, taking 43% of all images 
    of 1 and 2 in the MNIST training dataset
    """
    [x,y,input_shape]=Take_MNIST_Data(numbers_to_be_sent,percentage,one_hot)
    cl.x=x
    cl.y=y
    cl.imrow=input_shape[0]
    cl.imcol=input_shape[1]
    cl.num_channels=input_shape[2]
    cl.compile_neural_network
    #Reduce data if the particular client has to have lesser data
    slice_client_data(cl,cl.data_subset_size)
    
    cl.numbers_present=numbers_to_be_sent
    cl.one_hot=one_hot

def My_Shuffle(x,y,how_many_to_pick):
    #Shuffles the data 
    length=len(y)
    iterate=list(range(0,length))
    random.shuffle(iterate)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    tempx=x[indices]
    tempy=y[indices]
    return tempx[:how_many_to_pick], tempy[:how_many_to_pick]


#%% CREATE CLIENTS AND ASSIGN DATA TO EACH
def Create_Clients(number_of_clients,data_for_clients,In_Built_Training, percentage,client_percentages,**kwargs):
    """" 
    Create_Clients(how_many,data_for_clients,In_Built_Training):
    
    number_of_clients:          Positive Integer telling how many client objects to build
    data_for_clients:  An integer matrix where row i holds the MNIST samples to
                       be given to the ith client
    In_Built_Training: Boolean that enables or disables inbuilt training
    
    Returns:           Client objects with the data stored in them
    """
    #% GETTING DATA and LOSS FUNCTION
    batch_samples=kwargs.get('batch_samples',None)
    disjoint_subset=kwargs.get('disjoint_subset',False)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    skew_data=kwargs.get('skew_data',None)
    # logits= false implies the neural network output sums to 1. 
    #logits is a function that maps ptrobabilities to the real line
    one_hot=kwargs.get('one_hot',True)
    #%INITIALISING CLIENT OBJECTS
    cl=[]
    for i in range(number_of_clients):
        # temp=CC.Clients(loss_obj=loss_object, Weights=[],x=[],y=[])
        temp=Clients(loss_obj=loss_object,x=[],y=[])
        cl.append(temp)
        
    #%SETTING DATA in CLIENTS
    # data_for_clients=np.random.randint(9, size=[number_of_clients,4])
    for i in range(number_of_clients):
        cl[i].client_ID=i
        if disjoint_subset:
            #Get data only in the first iteration, don't repeatedly call the function every loop
            if i==0:
                [og_x,og_y,input_shape]=Take_MNIST_Data([0,1,2,3,4,5,6,7,8,9],percentage,one_hot)
                [new_x,new_y]=slice_tensors(og_x,og_y,number_of_clients) #Split into 'number_of_clients' disjoint batches
            cl[i].imrow=input_shape[0]
            cl[i].imcol=input_shape[1]
            cl[i].num_channels=input_shape[2]
            cl[i].compile_neural_network
            cl[i].data_subset_size=client_percentages[i] #Percentage of the disjoint MNIST Dataset to be in each client
            x=new_x[i].numpy()
            y=new_y[i].numpy()
            if one_hot:
                y=np.argmax(y,axis=1)
            [tempx,tempy,_]=Take_MNIST_Data(data_for_clients[i][:],client_percentages[i],one_hot,x=x,y=y,from_MNIST_Database=False)
            cl[i].x=tempx
            cl[i].y=tempy
            #Reduce data in each client to client_percentages[i]
            # slice_client_data(cl[i],cl[i].data_subset_size)
            cl[i].numbers_present=np.unique((np.argmax(cl[i].y.numpy(),axis=1)))
            cl[i].one_hot=one_hot
        else:
            cl[i].data_subset_size=client_percentages[i]
            Initialise_Client_Data(cl[i],data_for_clients[i][:],percentage,one_hot)
        
        #If data needs to be skewed Skew data should be an array of length equal to the number of classes, summing to 1 and positve
        if (type(skew_data)!=type(None)):
            if len(cl[i].numbers_present)!=len(skew_data[i]) or np.argmin(skew_data[i])<0:
                raise ValueError("Invalid skew data array (length is not the same as number of classes or there is a negative number)")
            index_of_classes=np.argmax(cl[i].y.numpy(),axis=1) if one_hot else cl[i].y.numpy() #Convert onehot to integer
            samples_per_class=[np.sum(index_of_classes==j)for j in cl[i].numbers_present]
            new_samples_per_class=np.int32(np.floor(np.asarray(skew_data[i])*samples_per_class))
            
            #Check if the skewness percentage required can be satisfied (There are enough samples in the first place)
            if (np.min(samples_per_class-new_samples_per_class)<0):
                print("Samples per class:",samples_per_class)
                print("Requested samples per class:",new_samples_per_class)
                raise ValueError("Not enough data samples present in some classes to skew at the given percentage")
            x=[]
            y=[]
            class_index=[]
            new_samples_per_class[new_samples_per_class==1]=2#If there is only 1 sample, put 2 instead Helps with train and test split later.
            if dataset_ID=='0' and iid_or_not=='n':
                # new_samples_per_class[new_samples_per_class>70]=70#Don't have more than 80 samples per class for MNIST, otherwise Local training is more practical
                new_samples_per_class=new_samples_per_class#Don't have more than 80 samples per class for MNIST, otherwise Local training is more practical
            elif dataset_ID=='1' and iid_or_not=='n':
                new_samples_per_class=new_samples_per_class# new_samples_per_class[new_samples_per_class>70]=70#Don't have more than 80 samples per class for CIFAR, otherwise Local training is more practical
            data_index=tf.range(start=0,limit=cl[i].x.numpy().shape[0],dtype=tf.int32).numpy()
            itr=0
            for j in cl[i].numbers_present:
                temp_index=data_index[index_of_classes==j] #Get the indicies of class j
                temp_index=temp_index[:new_samples_per_class[itr]] #Keep only 'new_samples_per_class[j]' number of samples
                class_index.append(temp_index)
                itr+=1
            temp=[]
            class_index=np.concatenate(class_index,axis=0)
            class_index=tf.constant(class_index, dtype=tf.int32) #Convert it into a tensor
            #Gather only the indexes that matter
            # zz=cl[i].x[class_index,:,:,:]
            cl[i].x=tf.gather(cl[i].x,class_index,axis=0)
            cl[i].y=tf.gather(cl[i].y,class_index,axis=0)
            
            cl[i].og_x=cl[i].x+0
            cl[i].og_y=cl[i].y+0
            cl[i].Shuffle_Tensor_Data()
            index_of_classes=np.argmax(cl[i].y.numpy(),axis=1) if one_hot else cl[i].y.numpy() #Convert onehot to integer
    
    #% TRAIN   
    if In_Built_Training:
        print("#########Beginning in-built training of clients ######### \n")
        for i in range(number_of_clients):
            print("Training client: ",i,"\n")
            cl[i].model.fit(x=cl[i].x,y=cl[i].y, epochs=10,batch_size=64,workers=4,use_multiprocessing=True,
                            verbose=False)
    return cl
#%% GET GRADIENTS
def Get_Grad_Split(obj, single_bit, query_weights,shared,**kwargs):
        """"
        Get_Grad(obj,single_bit) 
        returns signed gradient or normal gradient of obj, if single_bit is 
        true or false respectively
        """
        if shared==True:
            obj.model.layers[0].trainable=True
            obj.model.layers[1].trainable=True
            obj.model.layers[2].trainable=True
            obj.model.layers[3].trainable=True
            obj.model.layers[4].trainable=False
            obj.model.layers[5].trainable=False
            obj.model.layers[6].trainable=False
            obj.model.layers[7].trainable=False
        else:
            obj.model.layers[0].trainable=False
            obj.model.layers[1].trainable=False
            obj.model.layers[2].trainable=False
            obj.model.layers[3].trainable=False
            obj.model.layers[4].trainable=True
            obj.model.layers[5].trainable=True
            obj.model.layers[6].trainable=True
            obj.model.layers[7].trainable=True
        own_weights=obj.model.get_weights()
        update_batch_index=kwargs.get('update_batch_index',True)#If true, batch index is incremented here.
        if query_weights==None:
           query_weights=own_weights #Gradient at current weights.
        
        obj.model.set_weights(query_weights)
        def loss(obj,**kwargs):
          dropout=kwargs.get('dropout',True)
          # y_=obj.model(obj.x, training=True)
          obj.which_batch=obj.batch_index-1 if obj.which_batch<=obj.batch_size else 0
          #np.random.randint(low=0,high=obj.batch_size) #Which batch to use
          y_=obj.model(obj.x_batch, training=dropout)#training=True means that dropout will be in effect
          return obj.loss_obj(y_true=obj.y_batch, y_pred=y_)
        #% GRADIENT
        def grad(obj):
          with tf.GradientTape() as tape:
            loss_value = loss(obj,dropout=False)
            true_loss=loss(obj,dropout=False)
          return true_loss, tape.gradient(loss_value, obj.model.trainable_variables)
        #% CALLING EVERYTHING
        loss_value, grads = grad(obj)
        if update_batch_index:
            obj.batch_index+=1 #Increment batch index to get gradient using data from the next batch for the next iteration.
        my_grad=[]
        for i in range(len(grads)):
            my_grad.append(grads[i].numpy()) #Grads is a list tensor containing the gradients at each layer. This Converts it to a single list 
        
        if shared==True:
            obj.model.layers[0].trainable=False
            obj.model.layers[1].trainable=False
            obj.model.layers[2].trainable=False
            obj.model.layers[3].trainable=False
            obj.model.layers[4].trainable=True
            obj.model.layers[5].trainable=True
            obj.model.layers[6].trainable=True
            obj.model.layers[7].trainable=True
        else:
            obj.model.layers[0].trainable=True
            obj.model.layers[1].trainable=True
            obj.model.layers[2].trainable=True
            obj.model.layers[3].trainable=True
            obj.model.layers[4].trainable=False
            obj.model.layers[5].trainable=False
            obj.model.layers[6].trainable=False
            obj.model.layers[7].trainable=False
        #Putting back original weights after computing gradient.
        obj.model.set_weights(own_weights)
        #% Return signed gradient or whole gradient
        if single_bit:
            new_grad=my_grad
            itr=0
            for temp in my_grad:
                new_grad[itr]=np.int8(np.sign(temp))
                itr=itr+1
            return loss_value, new_grad
        else:
            return loss_value, my_grad
#%% GET GRADIENTS
def Get_Grad(obj, single_bit, query_weights,**kwargs):
        """"
        Get_Grad(obj,single_bit) 
        returns signed gradient or normal gradient of obj, if single_bit is 
        true or false respectively
        """
        obj.model.layers[0].trainable=True
        obj.model.layers[1].trainable=True
        obj.model.layers[2].trainable=True
        obj.model.layers[3].trainable=True
        obj.model.layers[4].trainable=True
        obj.model.layers[5].trainable=True
        obj.model.layers[6].trainable=True
        obj.model.layers[7].trainable=True
        own_weights=obj.model.get_weights()
        update_batch_index=kwargs.get('update_batch_index',True)#If true, batch index is incremented here.
        if query_weights==None:
           query_weights=own_weights #Gradient at current weights.
        
        obj.model.set_weights(query_weights)
        def loss(obj,**kwargs):
          dropout=kwargs.get('dropout',True)
          # y_=obj.model(obj.x, training=True)
          obj.which_batch=obj.batch_index-1 if obj.which_batch<=obj.batch_size else 0
          #np.random.randint(low=0,high=obj.batch_size) #Which batch to use
          y_=obj.model(obj.x_batch, training=dropout)#training=True means that dropout will be in effect
          return obj.loss_obj(y_true=obj.y_batch, y_pred=y_)
        #% GRADIENT
        def grad(obj):
          with tf.GradientTape() as tape:
            loss_value = loss(obj,dropout=False)
            true_loss=loss(obj,dropout=False)
          return true_loss, tape.gradient(loss_value, obj.model.trainable_variables)
        #% CALLING EVERYTHING
        loss_value, grads = grad(obj)
        if update_batch_index:
            obj.batch_index+=1 #Increment batch index to get gradient using data from the next batch for the next iteration.
        my_grad=[]
        for i in range(len(grads)):
            my_grad.append(grads[i].numpy()) #Grads is a list tensor containing the gradients at each layer. This Converts it to a single list 
        
        #Putting back original weights after computing gradient.
        obj.model.set_weights(own_weights)
        #% Return signed gradient or whole gradient
        if single_bit:
            new_grad=my_grad
            itr=0
            for temp in my_grad:
                new_grad[itr]=np.int8(np.sign(temp))
                itr=itr+1
            return loss_value, new_grad
        else:
            return loss_value, my_grad
#%% REDUCING CLIENT DATA
"""This function reduces the amount of data stored in a client object by subset%
eg.) If a client has 100 data samples as a tensor, if subset=0.2, 
this function will make it 20 data samples."""
def slice_client_data(obj,subset):
    #Cuts the data in a tensor
    [how_many_samples,imrow,imcol,rgb_chan]=tf.shape(obj.x).numpy()
    [_,how_many_classes]=tf.shape(obj.y).numpy()
    
    #Flooring the amount to be reduced
    reduce_to=int(np.floor(how_many_samples*subset))
    #tf.slice(tensor, [x-dim,y-dim],[x-dim-end,y-dim-end])
    
    new_x=tf.slice(obj.x,[0,0,0,0],[reduce_to,imrow,imcol,rgb_chan])
    new_y=tf.slice(obj.y,[0,0],[reduce_to,how_many_classes])
    obj.x=new_x
    obj.y=new_y
    
#%% QUERY GRADIENT MATRIX        
def Query_Grads_and_Losses_Split(obj,W,shared,dataset_ID,**kwargs):
    """
    This function returns the gradient of obj with respect to all the 
    weights present in W. single bit controls if either a quantised or true gradient 
    value is returned
    If mentioned, subset, slices the data contained in the object and the gradient is 
    computed with the smaller dataset, after which the original data is placed back 
    in the client object.
    """
    single_bit=kwargs.get('single_bit',True)         # If it isn't mentioned, it's taken as true
    single_client=kwargs.get('single_client',False)
    quiet=kwargs.get('quiet',False) #Verbose On or off
        
#GRADIENT AND LOSS COMPUTATION
    if not single_client:
        [number_of_clients,_,_]=np.shape(W)
    else:
        #Otherwise np.shape gives a scalar value and it throws an error
        number_of_clients=1         
    g=[]
    Loss_Value=np.zeros(number_of_clients)
    for i in range(number_of_clients):
        if number_of_clients>1:        
            w=De_Vectorize_W(W[i],obj)
        else:
            w=De_Vectorize_W(W,obj)
        
        obj.model.set_weights(w)
        [Loss_Value[i], temp]=Get_Grad_Split(obj,single_bit,w,shared,update_batch_index=False)
        if dataset_ID=='0': 
             temp2 = [tf.zeros([3, 3, 1, 28], tf.int32),tf.zeros([28,], tf.int32),tf.zeros([4732, 128], tf.int32),tf.zeros([128,], tf.int32),temp[0],temp[1]]
        elif dataset_ID=='1':
            temp2 = [tf.zeros([3, 3, 1, 32], tf.int32),tf.zeros([32,], tf.int32),tf.zeros([5408, 128], tf.int32),tf.zeros([128,], tf.int32),temp[0],temp[1]]
#        temp2.append(0)
#        temp2.append(0)
#        temp2.append(0)
#        temp2.append(0)
#        wei = obj.model.layers[0].get_weights()
#        print(wei[0].get_shape())
#        for x in range(0,8):
#            try:
#                wei = obj.model.layers[i].get_weights()
#                print(wei.get_shape())
#            except:
#                print("oops")
#                pass
        # check this
        if shared==True:
            temp.append(tf.zeros([128, 10], tf.int32))
            temp.append(tf.zeros([10, ], tf.int32))
#        else:
#            temp2[0]=tf.zeros([3, 3, 1, 28], tf.int32)
#            temp2[1]=tf.zeros([28,], tf.int32)
#            temp2[2]=tf.zeros([4732, 128], tf.int32)
#            temp2[3]=tf.zeros([128,], tf.int32)
#            temp2[4]=temp[0]
#            temp2[5]=temp[1]
#            temp2[4]=tf.zeros([128,10], tf.int32)
#            temp2[5]=tf.zeros([10,], tf.int32)
        #setting object weights as the computed gradient
        if shared==True:
            obj.model.set_weights(temp) 
        else:
            obj.model.set_weights(temp2) 
        #Calling the vectorize function to get the weights in vector form
        [temp,_]=Vectorize_W(obj)
        temp=np.int8(temp) if single_bit else temp
        g.append(np.squeeze(temp))
        if not quiet:
            print("Computed Gradient w.r.t weights", i)
#        print("Length!!!")
#        print(len(Loss_Value))
    if single_client:
        return Loss_Value[0], g[0]
    else:
        return Loss_Value, g

#%% QUERY GRADIENT MATRIX        
def Query_Grads_and_Losses(obj,W,**kwargs):
    """
    This function returns the gradient of obj with respect to all the 
    weights present in W. single bit controls if either a quantised or true gradient 
    value is returned
    If mentioned, subset, slices the data contained in the object and the gradient is 
    computed with the smaller dataset, after which the original data is placed back 
    in the client object.
    """
    single_bit=kwargs.get('single_bit',True)         # If it isn't mentioned, it's taken as true
    single_client=kwargs.get('single_client',False)
    quiet=kwargs.get('quiet',False) #Verbose On or off
        
#GRADIENT AND LOSS COMPUTATION
    if not single_client:
        [number_of_clients,_,_]=np.shape(W)
    else:
        #Otherwise np.shape gives a scalar value and it throws an error
        number_of_clients=1         
    g=[]
    Loss_Value=np.zeros(number_of_clients)
    for i in range(number_of_clients):
        if number_of_clients>1:        
            w=De_Vectorize_W(W[i],obj)
        else:
            w=De_Vectorize_W(W,obj)
        
        obj.model.set_weights(w)
        [Loss_Value[i], temp]=Get_Grad(obj,single_bit,w,update_batch_index=False)
#        print(temp)
        #setting object weights as the computed gradient
        obj.model.set_weights(temp) 
        #Calling the vectorize function to get the weights in vector form
        [temp,_]=Vectorize_W(obj)
        temp=np.int8(temp) if single_bit else temp
        g.append(np.squeeze(temp))
        if not quiet:
            print("Computed Gradient w.r.t weights", i)
#    print(Loss_Value)
    if single_client:
        return Loss_Value[0], g[0]
    else:
        return Loss_Value, g

#%%  GRADIENT ASCENT OR DESCENT
def grad_des_asc(w_old,gradient,eta,sign):
    #if sign==1, perform ascent, else descent
    eta=eta*sign
    w_old=np.squeeze(w_old)
    gradient=np.squeeze(gradient)
    w_new=w_old+eta*gradient
    return w_new
#%% OPTIMISER (Learning rate)
""" Implementing adaptive learning rates to simulate different optimizers """
def optimiser(server,grads,alpha_or_weights,**kwargs):
    learning_rate=kwargs.get('rate',0.1)
    
    if alpha_or_weights=='weights':
        v_old=server.vw
        del_W=np.squeeze(np.asarray(server.Weights)-np.asarray(server.W_oldw))
        D_old=server.D_w
    else:
        v_old=server.va
        del_W=server.Alpha_matrix-server.W_olda
        D_old=server.D_a
    #If there is only one client
    if server.number_of_clients==1:
        grads=np.squeeze(grads)
        
    algorithm=kwargs.get('algorithm','none')
    
    if algorithm.lower()=='rmsprop':
        """Pass alpha and beta values if you want custom values for them """
        alpha=kwargs.get('alpha',0.1)
        beta=kwargs.get('beta',0.9)
        epsilon=kwargs.get('epsilon',1e-6)
        v_new=(v_old*beta)   +    (1-beta)*(grads**2)
        lrng_rte=alpha/np.sqrt(epsilon+v_new)
        D_new=0
    
    elif algorithm.lower()=='adagrad':
        """Pass alpha and beta values if you want custom values for them """
        alpha=kwargs.get('alpha',0.001)
        epsilon=kwargs.get('epsilon',1e-6)
        v_new=v_old  +    (grads**2)
        lrng_rte=alpha/np.sqrt(epsilon+v_new)
        D_new=0
        
    elif algorithm.lower()=='momentum':
        """Pass beta value if you want a custom value for it"""
        beta=kwargs.get('beta',0.9)
        v_new=(v_old*beta)   +    (1-beta)*grads
        lrng_rte=v_new
        D_new=0

    elif algorithm.lower()=='adadelta':
        """Pass beta value if you want a custom value for it """
        # alpha=kwargs.get('alpha',0.001)
        beta=kwargs.get('beta',0.95)
        epsilon=kwargs.get('epsilon',1e-6)
        v_new=(v_old*beta)   +    (1-beta)*(grads**2)
        D_new=(D_old*beta)   +    (1-beta)*(del_W**2)
        lrng_rte=np.sqrt(epsilon+D_new)/np.sqrt(epsilon+v_new)
    else:#Constant rate
        return learning_rate
### END of algorithms
#Update parmeters
    if alpha_or_weights=='weights':
        server.vw=v_new
        server.W_oldw=server.Weights
        server.D_w=D_new
    else:
        server.va=v_new
        server.W_olda=server.Alpha_matrix
        server.D_a=D_new
#Return learning rate    
    return lrng_rte         

#%% L1 PROJECTION
#This functions finds the closest(Euclidean Distance) L1 unit sphere vector, x, to W
def project_L1(W):
    """ optimal x=sgn{w}*max{(abs(w)-u),0}
    Find u via Newton raphson method. More info can be found here:
    https://math.stackexchange.com/questions/2327504"""
    #first find u. initialise a point, randomly
    u_old=0
    u_new=0
    
    # x_old=np.sign(W)*np.maximum(abs(W)-u_old,0)
    for i in range(1000):
        temp=(abs(W)-u_old)
        temp1=temp
        temp1[temp1<=0]=0
        objective=np.sum(temp1)-1 
        temp[temp>0]=1
        derivative_u=np.sum(-temp)
               
        if derivative_u==0:
            x_new=np.sign(W)*np.maximum(abs(W)-u_old,0)
            break
        u_new=u_old-(objective/derivative_u)   #Newton Raphson
       
        if np.sum(abs(u_new-u_old))<1e-14:
            break
        u_old=u_new
    
    u_new=max(u_new,0)
    x_new=np.sign(W)*np.maximum(abs(W)-u_new,0)    
        
    return x_new

#%% QUERY CLIENT GRADIENTS AND LOSSES
def Query_Clients(server,W_query,single_bit):
    number_of_clients=server.number_of_clients
    Total_grad=[0]*number_of_clients
    Total_loss=[0]*number_of_clients
    #Querying each client for the gradients with respect to the broadcasted W matrix
    # print("Querying Clients For Gradients")
    for i in range(number_of_clients):
        server.cl[i].batch_index+=1 #Update batch index
        #Get loss and grads wrt each weight. temp_loss is a column vector
        old_wts=server.cl[i].model.get_weights() #Save weights before computing gradient
        [temp_loss, temp_grad]=Query_Grads_and_Losses(server.cl[i],W_query,quiet=True,single_bit=single_bit)
        server.cl[i].model.set_weights(old_wts) #Restore old weights after gradient computation
        Total_grad[i]=np.asarray(temp_grad) if single_bit else np.asarray(temp_grad)
        Total_loss[i]=temp_loss
#        print(len(Total_loss[i]))
#        if i%10==0:
#            print("Client ",i," gradients and losses ready")
    gc.collect()
#    print("Total loss length")
#    print(len(Total_loss))
    #Total_grad=np.asarray(Total_grad)
    return Total_grad, Total_loss #Row i contains loss/gradients computed in client i, column contains them wrt weights of client j

#%% QUERY CLIENT GRADIENTS AND LOSSES
def Query_Clients_Split(server,W_query,single_bit,shared,dataset_ID):
    number_of_clients=server.number_of_clients
    Total_grad=[0]*number_of_clients
    Total_loss=[0]*number_of_clients
    #Querying each client for the gradients with respect to the broadcasted W matrix
    # print("Querying Clients For Gradients")
    for i in range(number_of_clients):
        server.cl[i].batch_index+=1 #Update batch index
        #Get loss and grads wrt each weight. temp_loss is a column vector
        old_wts=server.cl[i].model.get_weights() #Save weights before computing gradient
        [temp_loss, temp_grad]=Query_Grads_and_Losses_Split(server.cl[i],W_query,shared,dataset_ID,quiet=True,single_bit=single_bit)
        server.cl[i].model.set_weights(old_wts) #Restore old weights after gradient computation
        Total_grad[i]=np.asarray(temp_grad) if single_bit else np.asarray(temp_grad)
        Total_loss[i]=temp_loss
#        print(len(Total_loss[i]))
#        if i%10==0:
#            print("Client ",i," gradients and losses ready")
    gc.collect()
    #Total_grad=np.asarray(Total_grad)
#    print("Total loss length")
#    print(len(Total_loss))
    return Total_grad, Total_loss #Row i contains loss/gradients computed in client i, column contains them wrt weights of client j

#%% COMPUTE W GRADIENTS
def get_W_grad(server):
    """This function computes the gradient of the weights. This is to be used in updating
    the query weight points
    """
    # print("Entered W function")
    shape_of_grads=np.shape(server.grad_W) #Remove singleton dimension
    gamma=server.gamma#/np.sqrt(itr+1) #Regularizer term for the loss function
    [num_clients,_,lamb_vector]=server.get_parameters
    q=len(server.cl[0].vector_weights) #The length of the weight vector at each client
    #Reordering recieved gradients from clients
    if server.single_bit:
        reordered_G=np.zeros(shape_of_grads,dtype="int8")
    else:
         reordered_G=np.zeros(shape_of_grads)
    if server.number_of_clients==1:
        temp=np.reshape(reordered_G,((1,1,shape_of_grads[2])))
        reordered_G=temp
        temp=None
        gc.collect()
    reordered_G=np.transpose(server.grad_W,(1,0,2))
    #Delete server.grad_W as it is no longer needed and takes up a lot of memory
    # server.grad_W=None
    gc.collect()
    #Computing the gradients
    G=np.zeros((num_clients,q))
    for l in range(num_clients):
        alpha=server.Alpha_matrix[:,l]
        lamb=lamb_vector[l]
        cl_wts=server.cl[l].vector_weights #For regularizing
        temp1=lamb*np.transpose(reordered_G[l])@alpha
        temp2=gamma*cl_wts
        temp2=np.squeeze(temp2)
        G[l,:]=temp1+temp2
    temp=np.sum(G,axis=0)
    G=[temp for _ in G]
    G=np.asarray(G)
    if server.number_of_clients==1:
        G=np.expand_dims(G,axis=0) #Add empty dimension for 1 client case. Python is annoying.
    return G

#%% COMPUTE OMEGA GRADIENTS
def get_O_grad(server):
    """This function computes the gradient of the weights. This is to be used in updating
    the query weight points
    """
    # print("Entered W function")
    shape_of_grads=np.shape(server.grad_W) #Remove singleton dimension
#    gamma=server.gamma#/np.sqrt(itr+1) #Regularizer term for the loss function
    [num_clients,Alpha_matrix,lamb_vector]=server.get_parameters
    q=len(server.cl[0].vector_weights) #The length of the weight vector at each client
    #Reordering recieved gradients from clients
    if server.single_bit:
        reordered_G=np.zeros(shape_of_grads,dtype="int8")
    else:
         reordered_G=np.zeros(shape_of_grads)
    if server.number_of_clients==1:
        temp=np.reshape(reordered_G,((1,1,shape_of_grads[2])))
        reordered_G=temp
        temp=None
        gc.collect()
    reordered_G=np.transpose(server.grad_W,(1,0,2))
    #Delete server.grad_W as it is no longer needed and takes up a lot of memory
    # server.grad_W=None
    gc.collect()
    #Computing the gradients
    G=np.zeros((num_clients,q))
    for l in range(num_clients):
#        print("Ola")
        alpha=server.Alpha_matrix[:,l]
        #lamb=lamb_vector[l]
        #cl_wts=server.cl[l].vector_weights #For regularizing
        #temp1=lamb*np.transpose(reordered_G[l])@alpha
        temp1=np.transpose(reordered_G[l])@alpha
        #temp2=gamma*cl_wts
        #temp2=np.squeeze(temp2)
        # G[l,:]=temp1+temp2
#        print("OlaOla")
        G[l,:]=temp1
    if server.number_of_clients==1:
        G=np.expand_dims(G,axis=0) #Add empty dimension for 1 client case. Python is annoying.
    return G

#%% COMPUTE ALPHA GRADIENTS
def get_alpha_grad(server):
    [num_clients,Alpha_matrix, disc_matrix,
                     lamb_vector]=server.get_parameters
    lamb=np.diag(lamb_vector)
    #Reordering recieved gradients from clients
    reordered_L=np.transpose(server.Losses)
    #Computing the gradients
    #Square root Denominator Term
    temp=0
    for j in range(num_clients):
        alj=Alpha_matrix[j,:]
        temp+=(np.dot(alj,lamb_vector)/server.cl[j].num_samples)**2
    term1=server.M*np.sqrt(np.log(1/server.delta_mcdiarmid)/(2*temp)) #Constant term
    # Square root numerator term
    temp=0
    term2=np.ones((num_clients,num_clients))
    for j in range(num_clients):
        alj=Alpha_matrix[j,:]
        #Client-wise info, in each row
        term2[j,:]=np.dot(alj,lamb_vector)/(server.cl[j].num_samples**2)*np.asarray(lamb_vector)
    tuning_param=1
    term3=term1*term2
    term_l2=2*abs(Alpha_matrix)*0
    return lamb@(reordered_L+disc_matrix)+term3+term_l2#Matrix multiplication
#%% ALPHA CVX
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import SR1
def alpha_cvx(server):
    p=server.number_of_clients
    print("Finding Alphas")
    for k in range(p):
        constraint=LinearConstraint(np.ones(p), lb=1, ub=1) #Equality constraint
        bounds = [(0, n) for n in np.ones(p)] #Inequality Constraint [alphas lie between 0 and 1]
        server.client_index_minimization=k
        f=server.loss_wrt_alpha
        # tesss=f(server.Alpha_matrix[k])
        x0=np.ones((p))*(1/p)
        temp=minimize(f, x0, bounds=bounds,constraints=constraint,method='trust-constr',  
                      jac="2-point", hess=SR1())
        # temp=minimize(f, x0, bounds=bounds,constraints=constraint,method='trust-constr',  jac="2-point")
        server.Alpha_matrix[:,k]=temp.x
        print("Done for client ",k)


#%% ALPHA CONSTRAINT PROJECTION
# Using the cvx toolbox, the alpha vector is projected onto the constraint set here.
""" j indicates which column of the alpha matrix is being projected"""
def Single_Alpha_Projection(server,alpha,j):    
    from cvxopt import matrix, solvers
    # from numpy import array
    p=server.number_of_clients
    #Main function
    Q = matrix(np.eye(p))
    c = -2*matrix(alpha) #Lies within the L2 ball
             
    #Equality constraint
    temp_kj=np.ones((1,p)) #Refering to all other elements of alpha
    ############BE OBSERVANT HERE. USE np.r_ TO SEND A MATRIX OF NP ARRAYS##########
    #IT DOESN'T WORK WITHOUT np.r_
    A=matrix(np.r_[temp_kj])
    b=matrix(1.0)
    
    #Inequality constraints
    lmbda=np.asarray(server.lamb_vector)
    # temp1=lmbda*np.ones((1,p))
    temp2=-np.eye(p)
    G=matrix(np.r_[temp2])
    temp4=np.asarray(np.zeros((p,)))
    h=matrix(np.r_[temp4])
    solvers.options['feastol']=1e-9
    solvers.options['show_progress']=False
    sol=solvers.qp(Q,c,G,h,A=None,b=None)
    ff=sol['x']
    ff=ff/np.sum(ff)
    # print (ff)
    return np.asarray(ff)
#%% PROJECT ALPHA TO CONSTRAINT (CVX TOOLBOX)
def project_alpha_to_constraint(server,Alpha_matrix):
    if server.number_of_clients==1:
        return np.asarray([[1.0]])
    projected_alpha=Alpha_matrix*0
    for j in range(server.number_of_clients):
        temp=Single_Alpha_Projection(server,Alpha_matrix[:,j],j)
        projected_alpha[:,j]=np.squeeze(temp)
    return projected_alpha
#%% MAIN LOSS FUNCTION
def main_loss(server):
    loss=0
    term1=0
    term2=0
    term3=0
    term4=0
    for k in range(server.number_of_clients):
        cl_wts=server.cl[k].vector_weights #For regularizing
        temp1=server.lamb_vector[k]*np.dot(server.Alpha_matrix[:,k],server.Losses[k])#server.cl[k].num_samples
        temp2=server.gamma*np.linalg.norm(cl_wts)
        temp3=server.lamb_vector[k]*np.dot(server.Alpha_matrix[:,k],server.disc_matrix[k,:])
        
        term1=term1+temp1 #Federated loss
        term2=term2+temp2 #regularizer
        term3=term3+temp3 #discrepancy and alpha
        loss=loss+temp1+temp2+temp3
    temp4=0
    #Square root term
    for j in range(server.number_of_clients):
        nj=server.cl[j].num_samples #number of samples in a client
        temp4=(np.dot(server.Alpha_matrix[j,:],server.lamb_vector)/nj)**2+temp4
    temp4=np.sqrt(temp4*np.log(1/server.delta_mcdiarmid)*0.5)
    term4=temp4*server.M
    loss=loss+temp4*server.M
    return loss,term1,term2,term3,term4
#%%Accuracy
"This function returns the average accuracies of all clients as well as the accuracies of all clients on the train as well as test data"
def Accuracy(server,cl):
    one_hot=server.cl[0].one_hot
    number_of_clients=server.number_of_clients
    accuracies=np.zeros((number_of_clients,2))
    avg_accuracies_train=0
    avg_accuracies_test=0
    for i in range(number_of_clients):
        for x,y,name in zip([cl[i].x,cl[i].x_test],[cl[i].y,cl[i].y_test],["Train Set", "Test Set"]):
            # print(f"Testing client {i} on "+name)
            y_pred=cl[i].model.predict(x)
            if one_hot: #if one hot is true, convert it to integer
                y=tf.argmax(y, axis=1)
            y_pred=tf.argmax(y_pred, axis=1)
            acc=np.int8((y_pred-y).numpy())
            acc[acc!=0]=1
            num_samples_tested=y.shape[0]
            acc=(num_samples_tested-sum(acc))/num_samples_tested
            if name=="Train Set":
                avg_accuracies_train=(acc+i*avg_accuracies_train)/(i+1) #Running average of accuracies
                accuracies[i,0]=acc
            else:
                avg_accuracies_test=(acc+i*avg_accuracies_test)/(i+1) #Running average of accuracies
                accuracies[i,1]=acc
            # print("\n")
    # print("Finished Testing\n")
    
    return np.asarray(accuracies),[avg_accuracies_train,avg_accuracies_test]
#%% SLICE TENSORS
def slice_tensors(x,y,batch_size):
    """This function slices the data x and y into batch_size number of batches"""
    ####Shuffling the data given########
    #Create a tensor sequence to act as indices
    data_index=tf.range(start=0,limit=x.numpy().shape[0],dtype=tf.int32) 
    #Get the new shuffled indices
    shuffled_index=tf.random.shuffle(data_index)
    #Reconstruct x, and y in the new shuffled index way
    x=tf.gather(x,shuffled_index,axis=0) 
    y=tf.gather(y,shuffled_index,axis=0)
    
    extra_data=x.numpy().shape[0]%batch_size
    if extra_data !=0:
        #Discard any excess data so that the data can be split into even number of batches
        reduce_to=x.numpy().shape[0]-extra_data 
    else:
        reduce_to=x.numpy().shape[0]        
        
    imrow=x.numpy().shape[1]
    imcol=x.numpy().shape[2]
    rgb_chan=x.numpy().shape[3]
    how_many_classes=y.numpy().shape[1]
    #Slice x
    new_x=tf.slice(x,[0,0,0,0],[reduce_to,imrow,imcol,rgb_chan]) #Slice out extra data so it can be split into batches.
    #Slice y
    new_y=tf.slice(y,[0,0],[reduce_to,how_many_classes]) #Slice out extra data so it can be split into batches.        
    batches_of_y=tf.split(new_y,batch_size,axis=0)
    batches_of_x=tf.split(new_x,batch_size,axis=0)
    
    return batches_of_x,batches_of_y
#%% TEST PERFORMANCE (CONFUSION MATRIX)
def Confusion_Metric(server,**kwargs):
    """Plots confusion matricies if  returns=False
    datasets=[0,1,2] tests only on these numbers"""
    returns=kwargs.get('returns',False)
    datasets=kwargs.get('datasets',0)
    client_object_passed=kwargs.get('client_object_passed',False)
    dont_use_MNIST_data=kwargs.get('dont_use_MNIST_data',True)
    use_train_set=kwargs.get('use_train_set',False)
    verbose=kwargs.get("verbose",True)
    #if client object is passed and not a server object
    if client_object_passed:
        one_hot=server[0].one_hot
        number_of_clients=len(server)
        cl=server
    else:
        one_hot=server.cl[0].one_hot
        number_of_clients=server.number_of_clients
        cl=server.cl
    if not dont_use_MNIST_data and datasets!=0: #If you want to use raw MNIST data and not data locally present at each client
        [og_x,og_y,_]=Take_MNIST_Data(datasets,1,one_hot)
    num_samples=[]
    my_accuracies=[]
    # my_confusions=[]
    for i in range(number_of_clients):
        error=0
        if dont_use_MNIST_data: #using local data
            if client_object_passed: #if client object is passed instead of server
                if use_train_set or type(server[i].x_test)==type(None): #If we are testing with train set data points or the test set is not yet initialised
                    [og_x,og_y]=[server[i].og_x,server[i].og_y]
                else:
                    [og_x,og_y]=[server[i].x_test,server[i].y_test]
            else:
                if use_train_set: #If we are testing with train set data points
                    [og_x,og_y]=[server.cl[i].x,server.cl[i].y]
                else:                    
                    [og_x,og_y]=[server.cl[i].x_test,server.cl[i].y_test]
        if og_x.shape[0]>40000:
            [new_x,new_y]=slice_tensors(og_x,og_y,4) #Split into 4 batches so that large data sets can fit into GPU memory and run faster
        else:
            new_x=[og_x]
            new_y=[og_y]
        itr=0
        for x,y in zip(new_x,new_y):
            num_samples.append(x.shape[0])
            prediction=np.argmax(cl[i].model(x).numpy(),axis=1)
            true_label=np.argmax(y.numpy(),axis=1)
            if itr==0:
                confuse_temp=tf.math.confusion_matrix(true_label,prediction,num_classes=y.shape[1]).numpy()
            else:
                confuse_temp=tf.math.confusion_matrix(true_label,prediction,num_classes=y.shape[1]).numpy()+confuse_temp
            itr=itr+1
            # my_confusions.append(confuse_temp)
            error=confuse_temp+error
        
        acc=np.trace(error)/np.sum(error)
        my_accuracies.append(acc)
        if returns != True:
            print("Confusion Matrix for Client {} \n".format(cl[i].client_ID))
            print(confuse_temp)        
            print('\n')
    itr=0
    if not client_object_passed: #If server object is passed
        if use_train_set==True:
            num_samples=np.sum([cl[i].training_samples_per_class for i in range(len(server.cl))],axis=1)
        else:
            num_samples=np.sum([cl[i].testing_samples_per_class for i in range(len(server.cl))],axis=1)
        print("".center(100,'#'))
        # print(server.name.center(100,' '))
        print("".center(100,'#'))
    if verbose==True:
        
        for acc in my_accuracies:
            print("Client {} Accuracy--> {:0.2f}% ".format(cl[itr].client_ID,acc*100),"Samples Tested-->",num_samples[itr])
            itr=itr+1
    if returns:
        return np.asarray(my_accuracies)
#%% FINAL RESULTS
def Final_Results(server,weight_history,Loss_History,avg_acc):
  server.finalise_client_weights(server.Weights)
  # server.cl=cl
  after_alg_proposed_train=Confusion_Metric(server,returns=True,use_train_set=True)
  after_alg_proposed_test=Confusion_Metric(server,returns=True)

  print("".center(50,'-'))
  print("Alphas".center(50,' '))
  print("".center(50,'-'))
  # print(np.round(server.Alpha_matrix,3))

  print("".center(50,'-'))
  print("Discrepancy Values".center(50,' '))
  print("".center(50,'-'))
  # print(np.round(server.disc_matrix,3))

  #% GRAPHS
  import matplotlib.pyplot as plt
  #Weight Difference Plot
#  plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
#  # plt.rcParams.update({'font.size': 20})
#  plt.xlabel('Server-Client Calls',fontsize=20)
#  plt.ylabel('Norm Squared Difference',fontsize=20)
#
#  plot_title_name=server.name+' Norm-Squared Weight Difference between Iterations [Signed ' +server.optimiser+']' if server.single_bit else 'Norm-Squared Weight Difference between Iterations [Unsigned ' +server.optimiser+']'
#  plt.title(plot_title_name)
#  plt.plot(weight_history)
#  legend_name=["Client {}".format(i) for i in range(server.number_of_clients)]
#  # plt.legend(legend_name)
#  plt.show()
#  #Main Loss Plot
#  plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
#  # plt.rcParams.update({'font.size': 20})
#  plt.xlabel('Server-Client Calls',fontsize=20)
#  plt.ylabel('Loss Value',fontsize=20)
#
#  plot_title_name=server.name+' Main Loss [Signed ' +server.optimiser+']' if server.single_bit else 'Main Loss [Unsigned ' +server.optimiser+']'
#  plt.title(plot_title_name)
##  plt.plot(np.asarray(Loss_History)[:,[0]]) # check this
#  legend_name=["Main Loss", "Federated Loss", "Regularizer Loss","Alpha Discrepancy Loss","Square-root term loss"]
#  plt.legend(legend_name)
##  plt.show() # check this
#  temp=np.asarray(Loss_History)
#  m=np.mean(temp,axis=0)
#  std=np.std(temp,axis=0)
#  temp1=(temp-m)/std
#  temp1[:,2]=0
  
  
  plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
  plot_title_name=server.name+' Accuracies [Signed ' +server.optimiser+']' if server.single_bit else 'Main Loss [Unsigned ' +server.optimiser+']'
  plt.title(plot_title_name)
  plt.plot(np.asarray(avg_acc))
  legend_name=["Train Accuracy", "Test Accuracy"]
  plt.legend(legend_name)
  plt.xlabel('Server-Client Calls',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.show()

#  plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
#  plot_title_name=server.name+' Loss Components Trends (normalised) [Signed ' +server.optimiser+']' if server.single_bit else 'Loss Components Trends (normalised) [Signed  ' +server.optimiser+']'
#  plt.title(plot_title_name)
#  plt.plot(temp1)
#  legend_name=["Main Loss", "Federated Loss", "Regularizer Loss","Alpha Discrepancy Loss","Square-root term loss"]
#  plt.legend(legend_name)
  return after_alg_proposed_train,after_alg_proposed_test
#%% SLEEP
def finish():
    import time
    for _ in range(4):
        print('\a')
        time.sleep(1)
    return None
#%% CLIENT DATA AUGMENTATION
def augment_data(cl,**kwargs):
    aug_test_data=kwargs.get('aug_test_data',False)
    #Saving a copy of the original data in client object
    x_data=cl.x
    y_data=cl.y
    if aug_test_data==True:
            x_data=cl.x_test
            y_data=cl.y_test
    n=x_data.shape[0]
    #Create Altered Dataset
    radian=np.random.uniform(-np.pi/7 ,np.pi/7,(n,))#Rotate each image by a random angle
    position=np.random.uniform(-5 ,5,(n,2,)) #Move each image by a random position
    position=tf.convert_to_tensor(position, dtype=float) #Convert the position to a tensor, to send into function
    new=tfa.image.transform_ops.rotate(x_data,radian) #new is the rotated image set
    new=tfa.image.translate(new, position,'NEAREST') #new gets translated to new
    #Collecting them together
    new_x=tf.concat([new,x_data],axis=0) #Augmenting x data
    new_y=tf.concat([y_data,y_data],axis=0) #Augmenting y data    
    if aug_test_data==False:
        cl.x=new_x #Replace the x data
        cl.y=new_y #Replace the y data
    else:
        cl.x_test=new_x #Replace the x data
        cl.y_test=new_y #Replace the y data
#%% TRAIN-TEST SPLIT

def train_test_split(cl_main,**kwargs):
    #For some reason it won't work if I import this outised the function. I don't know why
    from sklearn.model_selection import train_test_split 
    """This function splits the data in each of the client objects into train and test sets 
    Returns an array holding number of training samples, number of samples/5,  testing samples and  number of samples"""
    test_ratio=kwargs.get('test_ratio',0.8)
    augmentation=kwargs.get('augmentation',False)
    returns=kwargs.get('returns',False)
    number_of_clients=len(cl_main)
    samples_per_class=[cl_main[i].samples_per_class for i in range(number_of_clients)]
    
    #Fill it up with the index of the samples in that class, present in og_x and og_y
    #Find the umber of test samples to take and split the index arrays
    #Fill up x and x_test
    for cl1 in cl_main:
        x=cl1.og_x.numpy()
        integer_labels=np.argmax(cl1.og_y.numpy(),axis=1)
        [x_train, x_test, y_train, y_test]=train_test_split(x,integer_labels,test_size=test_ratio,random_state=42,stratify=integer_labels)
        #Convert to categorical if needed
        if cl1.one_hot:
            y_train=to_categorical(y_train,cl1.num_classes)
            y_test=to_categorical(y_test,cl1.num_classes)
        cl1.x=tf.constant(x_train, dtype=tf.float32)
        cl1.y=tf.constant(y_train, dtype=tf.float32)
        cl1.x_test=tf.constant(x_test, dtype=tf.float32)
        cl1.y_test=tf.constant(y_test, dtype=tf.float32)
        


    ztrn=np.asarray([cl1.x.shape[0] for cl1 in cl_main]) #number of training
    ztst=np.asarray([cl1.x_test.shape[0] for cl1 in cl_main]) #number of testing
    zzz=[ztrn,np.floor((ztrn+ztst)/5),ztst,ztrn+ztst] #To compare number of samples
    zzz=np.asarray(zzz)
    if returns==True:
        return zzz.T
#%% SAVE DATA TO DRIVE
"""function to save the train and test data of clients onto drive"""
import os
def save_data(cl,path,**kwargs):
    name=kwargs.get('name','')
    p=len(cl)
    for client in cl:
        client_folder_path=path+str(client.client_ID)+'/'
        os.makedirs(client_folder_path,exist_ok=True)
        np.save(client_folder_path+'x_train',client.x,allow_pickle = False)
        np.save(client_folder_path+'y_train',client.y,allow_pickle = False)
        np.save(client_folder_path+'x_test',client.x_test,allow_pickle = False)
        np.save(client_folder_path+'y_test',client.y_test,allow_pickle = False)
        np.save(client_folder_path+'og_x',client.og_x,allow_pickle = False)
        np.save(client_folder_path+'og_y',client.og_y,allow_pickle = False)
        print(f"Saved "+name+f" Client {client.client_ID}'s data in "+client_folder_path)
    print('\n')
#%% LOAD DATA FROM DRIVE
"""function to load the train and test data of clients onto client objects"""
def load_data(cl,path,**kwargs):
    name=kwargs.get('name','')
    itr=0
    for _ in cl:
        client_folder_path=path+str(cl[itr].client_ID)+'/'
        os.makedirs(client_folder_path,exist_ok=True)
        cl[itr].x=tf.constant(np.load(client_folder_path+'x_train.npy'), dtype=tf.float32)
        cl[itr].y=tf.constant(np.load(client_folder_path+'y_train.npy'), dtype=tf.float32)
        cl[itr].x_test=tf.constant(np.load(client_folder_path+'x_test.npy'), dtype=tf.float32)
        cl[itr].y_test=tf.constant(np.load(client_folder_path+'y_test.npy'), dtype=tf.float32)
        cl[itr].og_x=tf.constant(np.load(client_folder_path+'og_x.npy'), dtype=tf.float32)
        cl[itr].og_y=tf.constant(np.load(client_folder_path+'og_y.npy'), dtype=tf.float32)
        print(f"Loaded "+name+f" Client {cl[itr].client_ID}'s data from "+client_folder_path)
        itr+=1
    print('\n')
#%% Final GRAPHS
import matplotlib.pyplot as plt
import matplotlib.markers as markers
def Graph_Me(loading_ID,start,iterations):
    x=range(start,iterations)
    path_to_save_load='Data/'
    # path_to_save_load='Graphs/'
    #% Plotting Accuracies
    THINGS_TO_PLOT=[
        "avg_acc_",
        "Local_Training_acc_batch_100_",
        "Local_Testing_acc_batch_100_",
        "avg_acc_vanilla_"
        ]
    NAMES=[
        'Train Accuracy Omni-Fedge',
        'Test Accuracy Omni-Fedge',
        'Local Train Accuracy (100 Batches)',
        'Local Test Accuracy (100 Batches)',
        'Train Accuracy FedSGD',
        'Test Accuracy FedSGD'
    ]
    itr=-1
    colors=['tab:blue','tab:orange','black','green']
    comm_rounds=0
    van_clr_indx = 1
    clr_indx=2
    # jet= plt.get_cmap('jet')
    # colors = iter(jet(np.linspace(0,1,10)))
    for thing in THINGS_TO_PLOT:
        itr+=1
        thing_path=path_to_save_load+loading_ID+"/Output_Text_Files/"+thing+loading_ID+".txt"
        try:
            loaded_thing=np.loadtxt(thing_path)
            plt.figure(num=1, figsize=(20, 20), dpi=200, facecolor='w', edgecolor='k')
            plt.yticks(np.arange(0,105,5),fontsize=20)
            plt.xticks(np.arange(start,iterations,50),fontsize=20)
#            if thing=='avg_acc_' or thing=='avg_acc_vanilla_':
            if thing=='avg_acc_':
                #Plot train accuracy
                plt.plot(x,loaded_thing[start:iterations,0]*100,color=colors[itr],label=NAMES[itr],dashes=[3, 5, 5, 10],linewidth=2,marker=".",markevery=20)
                #Plot test Accuracy
                plt.plot(x,loaded_thing[start:iterations,1]*100,color=colors[itr],label=NAMES[itr+1],linewidth=3,marker="*",markevery=20)
                comm_rounds=iterations
                plt.legend(loc="lower right")
                plt.xlabel("Communication Rounds",fontsize=22)
                plt.ylabel("Accuracy %",fontsize=22)
            elif thing=='avg_acc_vanilla_':
                #Plot train accuracy
                plt.plot(x,loaded_thing[start:iterations,0]*100,color=colors[van_clr_indx],label=NAMES[itr+1],dashes=[3, 5, 5, 10],linewidth=2,marker="s",markevery=20)
                #Plot test Accuracy
                plt.plot(x,loaded_thing[start:iterations,1]*100,color=colors[van_clr_indx],label=NAMES[itr+2],linewidth=3,marker="^",markevery=20)
                comm_rounds=iterations
                plt.legend(loc="lower right")
#                plt.xlabel("Communication Rounds",fontsize=22)
#                plt.ylabel("Accuracy %",fontsize=22)
            else:
                y=np.average(loaded_thing,axis=0)
                y=y*np.ones(len(x))
                if itr%2!=0:#check
                    plt.plot(x,y*100,color=colors[clr_indx],label=NAMES[itr+1],dashes = [3, 5, 5, 5],linewidth=2,marker="X",markevery=20)
                else:
                    plt.plot(x,y*100,color=colors[clr_indx],label=NAMES[itr+1],linewidth=2,marker="D",markevery=20)
                    clr_indx+=1
                plt.grid(b=True,linewidth=0.5,which='both')
                leg = plt.legend()
                leg.get_lines()[0].set_linewidth(10)
                plt.legend(fontsize=20)
                plt.xlabel("Communication Rounds")
                plt.ylabel("Accuracy %")
                plt.title('Average Accuracies (FMNIST)',fontsize=22)
        except (OSError,FileNotFoundError) as e:
            print(thing_path+' not found')
            pass
    plt.savefig(path_to_save_load+loading_ID+"/Accuracies")
    plt.close(fig=1)
#%                                              Plotting Losses and WEIGHTS
    
    THINGS_TO_PLOT=[
        "Loss_History_",
        "Loss_History_vanilla_",
        "weight_history_",
        "weight_history_vanilla_"
    ]
    NAMES=[
        'Loss History Omni-Fedge',
        'Loss History Fed SGD',
        'Norm Squared Difference of Weights Averaged (Omni-Fedge)',
        'Norm Squared Difference of Weights Averaged (FedSGD)'
    ]
    itr=-1
    comm_rounds=0
    # jet= plt.get_cmap('jet')
    # colors = iter(jet(np.linspace(0,1,10)))
    for thing in THINGS_TO_PLOT:
        itr+=1
        thing_path=path_to_save_load+loading_ID+"/Output_Text_Files/"+thing+loading_ID+".txt"
        try:
            loaded_thing=np.loadtxt(thing_path)
            print(loaded_thing[start:iterations,0])
            print(loaded_thing[start:iterations,1])
            plt.figure(num=None, figsize=(16, 9), dpi=220, facecolor='w', edgecolor='k')
            plt.xticks(np.arange(start,iterations,50))
            # plt.yticks(np.linspace(0,100,50))
            if thing=='Loss_History_vanilla_' or thing=='Loss_History_':
#                plt.plot(x,loaded_thing[start:iterations,0],label=NAMES[itr])
                y = np.average(loaded_thing[start:iterations],axis=1)
                plt.plot(x,y,label=NAMES[itr])
                print("skip")
            else:
                y=np.average(loaded_thing[start:iterations],axis=1)
                plt.plot(x,y,label=NAMES[itr])
            comm_rounds=iterations
            plt.legend()
            plt.grid(b=True,which='both')
            plt.xlabel("Communication Rounds",fontsize=22)
#            plt.ylabel(NAMES[itr],fontsize=22)
            plt.title(NAMES[itr],fontsize=25)
            plt.savefig(path_to_save_load+loading_ID+'/'+THINGS_TO_PLOT[itr])
        except (OSError,FileNotFoundError) as e:
            print(thing_path+' not found')
            pass