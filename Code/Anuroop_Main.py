# %tensorflow_version 2.x
import os
import tensorflow as tf
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import SR1
# gpus = tf.config.experimental.list_physical_devices('GPU')# Lists all available GPUs
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')# Forces it to use only gpu 0
# tf.config.experimental.set_memory_growth(gpus[0], True)# Forces Tensorflow to use as much memory as needed
programID=input("Enter a Program ID \n") #So that while saving files, this ID will be appended to the end of it
iid_or_not=input("Do you want IID data allocation? y/n \n")
iid_or_not=iid_or_not.lower() #Convert to lower case
start_with_local=input("Do you want to initialise training with locally trained weights? y/n \n")
start_with_local=iid_or_not.lower() #Convert to lower case

dataset_ID=input("Type 0 for MNIST \nType 1 for FASHION MNIST\n")
if dataset_ID=='0':
    percentage=input("There are 70,000 images in total for MNIST, what fraction of it do you want to use? (Advised 0.4) \n")
else:
    percentage=input("There are 60,000 images in total for FASHION MNIST, what fraction of it do you want to use? (Advised 0.5) \n")
percentage=float(percentage)#Convert string to float
batch_size=int(input("How many batches do you want for federated learning gradient descent?\n"))

save_check=input("Do you want to load variables? (y/n) \n")
save_load_none=2 if save_check=='y' else 1 # 1 for save variables; 2 for load variables; 3 for neither. Make this 3 if it is not used in colab
path_to_save_load='Data/'+programID+'/'
fast_run=1 if save_load_none==2 else 0  #1 skips discrepancy computation 0 computes discrepancy. Is always 1 if save_load_none==2
loading_ID='' if save_load_none!=2 else input("Enter a Loading ID of files you want to load \n")
load_path=path_to_save_load
if loading_ID!='':
    load_path='Data/'+loading_ID+'/'
text_save_path=path_to_save_load+'Output_Text_Files/'+'/'
graphs_save_path=path_to_save_load+'Graphs/'+'/'
os.makedirs(text_save_path,exist_ok=True) #make a directory to store all results in as a text file
os.makedirs(graphs_save_path,exist_ok=True) #make a directory to store all graphs
#%% MODULES
import Anuroop_Helper as hf
import numpy as np
#%% CLIENTS
data_for_testing=[0,1,2,3,4,5,6,7,8,9]
number_of_clients=5
data_for_clients=[[1,2,3,4,5,6,7,8,9,0]]*number_of_clients
number_of_clients=np.asarray(data_for_clients).shape[0]
number_of_classes=10
#%% INITIALISE DATA IN CLIENTS
p=number_of_clients
if iid_or_not=='n':
    skew_data=[np.random.uniform(low=0,high=0.5,size=[number_of_classes])for _ in range(number_of_clients)]    
    for i in range (number_of_clients):
#        number_of_classes_per_client=np.random.randint(7,11,number_of_clients)
        number_of_classes_per_client = np.full(shape=number_of_clients,fill_value=number_of_classes,dtype=np.int)
        print(number_of_classes_per_client)
        mask=np.zeros((number_of_clients,number_of_classes))
        itr=0
        for x in number_of_classes_per_client:
            mask[itr,0:x]=1
            np.random.shuffle(mask[itr,:])
            itr+=1
    skew_data=np.multiply(skew_data,mask)
else:
    skew_data=np.ones((p,number_of_classes))
client_percentages=np.ones(number_of_clients)#[0.5,0.5,0.5,0.5] #What percent of the available data should be in each client

cl_pre_trained=hf.Create_Clients(number_of_clients,
                                 data_for_clients,
                                 False,
                                 percentage,
                                 client_percentages,
                                 disjoint_subset=True,
                                 skew_data=skew_data,
                                 dataset_ID=dataset_ID)
cl=hf.Create_Clients(number_of_clients,
                     data_for_clients,
                     False,
                     percentage,client_percentages,
                     disjoint_subset=True,
                     skew_data=skew_data,
                     dataset_ID=dataset_ID)
# Vanilla FedSGD
cl_vanilla=hf.Create_Clients(number_of_clients, 
                             data_for_clients,
                             False,
                             percentage,
                             client_percentages,
                             disjoint_subset=True,
                             skew_data=skew_data,
                             dataset_ID=dataset_ID)
print("".center(100,'#'))
print("Created Clients".center(100,' '))
print("".center(100,'#'))
#%% Train-Test Split
if save_load_none==1: #Split into train-test and save
    hf.train_test_split(cl,returns=True,augmentation=False,test_ratio=5/9)
    hf.save_data(cl,path_to_save_load+'data/',name='MTFeeL') #save cl's data
    hf.load_data(cl_pre_trained,path_to_save_load+'data/',name="Local Training") #load that data into other clients
    hf.load_data(cl_vanilla,path_to_save_load+'data/',name="FedSGD") #load that data into other clients
else: #Load from folders
    hf.load_data(cl,load_path+'data/',name='MTFeeL') #load cl's data
    hf.load_data(cl_pre_trained,load_path+'data/',name="Local Training") #load that data into other clients
    hf.load_data(cl_vanilla,load_path+'data/',name="FedSGD") #load that data into other clients
    hf.save_data(cl,path_to_save_load+'data/',name='MTFeeL') #save cl's data to the same folder after loading it, possibly from another folder
    
train_samples=cl[0].x.shape[0]
test_samples=cl[0].x_test.shape[0]
details=["Train Samples per client ="+str(train_samples),
         "Test Samples per client ="+str(test_samples),
         "Batch size ="+str(cl[0].batch_size),
         "Number Of Clients "+str(number_of_clients)
    ]
np.savetxt(text_save_path+"Details_"+programID+".txt",details,fmt='%s')
# hf.train_test_split(cl_pre_trained, cl_pre_trained, cl_pre_trained,augmentation=True,returns=True)

path_MTFeeL=path_to_save_load+'MTFeeL/'
path_Vanilla=path_to_save_load+'Vanilla/'

#%% Count Number of samples in each
samples_in_each_client=[cl[i].samples_per_class for i in range(number_of_clients)]
training_samples_in_each_client=[cl[i].training_samples_per_class for i in range(number_of_clients)]
testing_samples_in_each_client=[cl[i].testing_samples_per_class for i in range(number_of_clients)]

np.savetxt(text_save_path+"total samples_in_each_client"+programID+".txt",samples_in_each_client,fmt='%d',delimiter='\t')
np.savetxt(text_save_path+"training_samples_in_each_client"+programID+".txt",training_samples_in_each_client,fmt='%d',delimiter='\t')
np.savetxt(text_save_path+"testing_samples_in_each_client"+programID+".txt",testing_samples_in_each_client,fmt='%d',delimiter='\t')
#%% LAMBDA 
"""This is initialised as the fraction of samples in every client"""
total_samples=0
Lamb=[0]*number_of_clients
for i in range(number_of_clients):
    temp=cl[i].num_samples
    # [temp]=np.shape(cl[i].y.numpy())
    total_samples=total_samples+temp
    Lamb[i]=temp
samples_per_client=Lamb
Lamb=[temp/total_samples for temp in Lamb]

# Lamb[1]=1
#%% STARTING WEIGHTS (Xavier Initiialisation)
W_old=[0]*number_of_clients #Empty List
for i in range(number_of_clients):
    W_old[i]=cl_pre_trained[i].vector_weights
#%% SERVER
#Create a server object
server=hf.Server(cl)
server_vanilla=hf.Server(cl_vanilla)
server.Alpha_matrix = np.zeros((number_of_clients, number_of_clients))
#print(server.Alpha_matrix)
#print(server_vanilla.Alpha_matrix)
#%% Initital gradients
# Get the loss and gradient info from each client
single_bit=False
p=number_of_clients
server.single_bit=single_bit
server_vanilla.single_bit=single_bit
server_vanilla.lamb_vector=np.ones(p)/p
server_vanilla.Alpha_matrix = np.eye(p)
#%% Load Weights from Drive
everything_loaded=0
try:
    """Try to load weights from the drive if they are there """
    server.Weights=np.load(path_to_save_load+"MTFeeL_W.npy")
    server.finalise_client_weights(server.Weights) #Initialise client weights
    print("Loaded Old weights from Drive")
    old_weights_flag=True
    everything_loaded=1
    #Loading earlier weights and losses
    Loss_History=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"Loss_History_"+programID+".txt")
    Loss_History=np.ndarray.tolist(Loss_History) #Convert to list so we can append things to it
    
    Loss_History_vanilla=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"Loss_History_vanilla_"+programID+".txt")
    Loss_History_vanilla=np.ndarray.tolist(Loss_History_vanilla) #Convert to list so we can append things to it
    
    weight_history=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"weight_history_"+programID+".txt")
    weight_history=np.ndarray.tolist(weight_history) #Convert to list so we can append things to it
    
    weight_history_vanilla=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"weight_history_vanilla_"+programID+".txt")
    weight_history_vanilla=np.ndarray.tolist(weight_history_vanilla) #Convert to list so we can append things to it
    
    avg_acc=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"avg_acc_"+programID+".txt")
    avg_acc=np.ndarray.tolist(avg_acc) #Convert to list so we can append things to it
    
    avg_acc_vanilla=np.loadtxt(path_to_save_load+'Output_Text_Files/'+"avg_acc_vanilla_"+programID+".txt")
    avg_acc_vanilla=np.ndarray.tolist(avg_acc_vanilla) #Convert to list so we can append things to it
    
    iterations=len(avg_acc) #reload last time's iterations
    print("Loaded Everything from Drive")
    everything_loaded=2
except (FileNotFoundError,OSError) as e:
    #If some are not there, then initailaise them both to W_old, the one found in the locally trained clients
    print("Initialising things from scratch")
    if  everything_loaded==1:
        check_continue=input("\n Weights loaded, some of the other files were missing, continue fresh with new weights? (y/n)")
        if check_continue=='n':
            import sys
            sys.exit("Incomplete file Load")
    server.Weights=W_old
    server_vanilla.Weights=W_old
    old_weights_flag=False
    
    Loss_History=[]
    Loss_History_vanilla=[]
    weight_history=[]
    weight_history_vanilla=[]
    avg_acc=[]
    avg_acc_vanilla=[]
    iterations=0

#%%Local Training
#Create clients
itr=0
if everything_loaded!=2: #Skip training if we are loading local accuracies
    print("\n Locally Training Clients")
    batch_size=server.cl[0].batch_size
#    for btchsze in [batch_size,1]: #Test training set on batch size of 1 and on batch size used in federated learing
    for btchsze in [batch_size]: #Test training set on batch size used in federated learing
        cl_pre_trained=hf.Create_Clients(number_of_clients, data_for_clients, False, percentage,
                                              client_percentages,disjoint_subset=True,skew_data=skew_data)
        for clients,cl_local in zip(cl,cl_pre_trained):
            """Put the same data in clients"""
            cl_local.x=clients.x
            cl_local.y=clients.y
            
            cl_local.x_test=clients.x_test
            cl_local.y_test=clients.y_test
            
            cl_local.og_x=clients.og_x
            cl_local.og_y=clients.og_y
            itr+=1
        #Train Clients   
        for i in range(number_of_clients):
            print("Training client: ",i,"\n")
            # change epochs to 100
            cl_pre_trained[i].model.fit(x=cl_pre_trained[i].x,y=cl_pre_trained[i].y, epochs=100,batch_size=btchsze,workers=4,
                                        use_multiprocessing=True,verbose=False)
        #% Test them
        print("".center(100,'_'))
        print(f" Training Accuracy Local (batch size={btchsze})".center(100,' '))
        print("".center(100,'_'))
        use_train_set=True
        performances=hf.Confusion_Metric(cl_pre_trained,returns=True,use_train_set=use_train_set,client_object_passed=True,verbose=False)
        np.savetxt(text_save_path+f"Local_Training_acc_batch_{btchsze}_"+programID+".txt",performances,fmt='%f',delimiter='\t')
        # hf.train_test_split(cll, cll, cll,augmentation=True)
        [print(x*100) for x in performances]
        print("".center(100,'_'))
        print(" Testing Accuracy Local".center(100,' '))
        print("".center(100,'_'))
        use_train_set=False
        performances=hf.Confusion_Metric(cl_pre_trained,returns=True,use_train_set=use_train_set,client_object_passed=True,verbose=False)
        np.savetxt(text_save_path+f"Local_Testing_acc_batch_{btchsze}_"+programID+".txt",performances,fmt='%f',delimiter='\t')
        [print(x*100) for x in performances]
else:
    print("Skipping local training as local training data was loaded from drive")
#%% Change Starting weights if requested
if start_with_local=='y':
    W_old=[0]*number_of_clients #Empty List
    for i in range(number_of_clients):
        W_old[i]=cl_pre_trained[i].vector_weights+0
    
    server.Weights=W_old
    server_vanilla.Weights=W_old
    server.finalise_client_weights(W_old)
    server_vanilla.finalise_client_weights(W_old)
#%% Initial Gradients
"""Reset Epochs and other counters in all clients of MTFeeL (After finding discrepancy) """
for clients in cl:
    clients.local_epoch=0 if fast_run==0 else -1#Resetting epochs for training algorithm
    clients.batch_index=0#Resetting epochs for training algorithm
    
if old_weights_flag: #If weights were loaded from drive
    #Query MTFeeL clients
    [C_Gs, C_Ls]=hf.Query_Clients(server,server.Weights,False)
    server.Losses=C_Ls
    server.grad_W=C_Gs
    #Query FedSGD clients
    [C_Gs, C_Ls]=hf.Query_Clients(server_vanilla,server_vanilla.Weights,single_bit)
    server_vanilla.Losses=C_Ls
    server_vanilla.grad_W=C_Gs
else: #If old weights are not there, then both FedSGD and MTFeeL are initialised to 
      #the same weights so the gradient will be the same, so compute it only once
    [C_Gs, C_Ls]=hf.Query_Clients(server,W_old,False)
#Set the recieved losses and grads in the server
    server.Losses=C_Ls
    server.grad_W=C_Gs
    server_vanilla.Losses=C_Ls
    server_vanilla.grad_W=C_Gs
    
#%% OMEGA_MATRIX    
# compute loss_matrix
loss_mat = []
for i in range(0, number_of_clients):
   loss_mat.append(server.Losses[i])

# compute omega_matrix
lambda_hyp = 0.5
constraint=LinearConstraint(np.ones(number_of_clients), lb=1, ub=1) # elements of minimizing vector should sum to 1
bounds = [(0.0001, n) for n in np.ones(number_of_clients)] # each element of minimizing vector should be strictly greater than 0
def objective_fn(x,loss_vector): # definition of objective function to be minimized
	objective = np.dot(loss_vector, x**lambda_hyp)-(lambda_hyp*np.log(np.prod(x)))
	return objective
   
omega_matrix = []
for i in range(0, number_of_clients):
    res=minimize(objective_fn,np.ones((number_of_clients))*(1/number_of_clients),loss_mat[i],bounds=bounds,constraints=constraint)
    omega_matrix.append(np.array(res.x))
    server.Alpha_matrix[:,i]=res.x
my_gamma = hf.get_O_grad(server)
#%% ITERATIONS
server.gamma=0
thresh_epsilon=0.00015
# while epsilon>1e-9 and iterations<100:
global_epoch=0
#number_of_iterations=2
#number_of_iterations=500
number_of_iterations=1000
# number_of_iterations=50

print("Starting Main Training \n")
loops=0
mw=0 #Momentum weights
beta=0.9
optimiser='sgd'
server.optimiser=optimiser
server_vanilla.optimiser=optimiser
lr=1
lrng_rt_alhpa=0.01
import gc
# C_Gs=C_Ls=W_old=None
gc.collect()
lrn_rt=[]
lrn_rt_van=[]
lrn_rt_w=[]
lrn_rt_al=[]

server.gamma=0
weight_criterion_vanilla=100#Keep this high so vanilla FL starts
server_vanilla.gamma=0
stop_vanilla=False
d=cl[0].model.count_params() #Number of parameters for the neural network
print(d)
#rate=number_of_clients/(number_of_iterations*d)**(1/2)
rate=100*number_of_clients/(number_of_iterations*d)**(1/2) # vary this (50-100)
server.M=5000
server_vanilla.M=0
#%% Weights Saving Function
"""Function to quickly save all the weights of the client objects """ 
def save_weights():
    MTFeeL_W=[0]*number_of_clients #Empty List
    FedSGD_W=[0]*number_of_clients #Empty List
    for i in range(number_of_clients):
        MTFeeL_W[i]=server.cl[i].vector_weights
        FedSGD_W[i]=server_vanilla.cl[i].vector_weights
    np.save(path_to_save_load+"MTFeeL_W",MTFeeL_W)
    np.save(path_to_save_load+"FedSGD_W",FedSGD_W)
    MTFeeL_W=None
    FedSGD_W=None
    gc.collect()
#%%
try:
    while iterations<number_of_iterations:
        print('\n ########## Iteration {} ##########'.format(iterations))    
        #<FedSGD>
        if weight_criterion_vanilla>thresh_epsilon:
            w_gradient_vanilla=hf.get_W_grad(server_vanilla)
        else:
            w_gradient_vanilla=np.multiply(w_gradient_vanilla,0) #Stop changing vanilla gradient weights if the main algorithm is finished
            stop_vanilla=True
#        learning_rate_vanilla=hf.optimiser(server_vanilla,w_gradient_vanilla,'weights',algorithm=optimiser,rate=rate)
        learning_rate_vanilla=rate
        lrn_rt_van.append(np.average(learning_rate_vanilla))
        old_wts_vanilla=np.squeeze(np.asarray(server_vanilla.Weights))
        """ FedSGD Gradient descent step"""
        if optimiser=='momentum':
            #learning rate itself is the gradient
            new_wts_vanilla=hf.grad_des_asc(server_vanilla.Weights,learning_rate_vanilla,lr,-1)
        else:
            new_wts_vanilla=hf.grad_des_asc(server_vanilla.Weights,w_gradient_vanilla,learning_rate_vanilla,-1) # <check the learning rate>
        temp_vanilla=np.linalg.norm(old_wts_vanilla-np.squeeze(np.asarray(new_wts_vanilla)),axis=1)
        weight_criterion_vanilla=np.average(temp_vanilla)
        weight_history_vanilla.append(temp_vanilla)
        gc.collect()
        #Converting the obtained weights to an appropriate format
        if server.number_of_clients==1:
            new_wts_vanilla=[np.expand_dims(new_wts_vanilla,axis=1)]
        else:
            new_wts_vanilla=[np.expand_dims(temp,axis=1) for temp in new_wts_vanilla]
        server_vanilla.Weights=new_wts_vanilla #New Query Weights
        server_vanilla.finalise_client_weights(server_vanilla.Weights) #Update learned weights to all clients
        print("\nQuerying clients for gradients (Vanilla Server)")
#        C_Ls_van=None
#        C_Gs_van=None
        if not stop_vanilla:
            [C_Gs_van, C_Ls_van]=hf.Query_Clients(server_vanilla,server_vanilla.Weights,single_bit)#Single bit True or False. Query the gradients and losses at all clients
            server_vanilla.Losses=C_Ls_van
            server_vanilla.grad_W=C_Gs_van
        else: #Training has stopped, so no need to query
#            C_ls_van=server_vanilla.Losses
#            C_Gs_van=server_vanilla.grad_W
            print("Vanilla training done")
#        server_vanilla.Losses=C_Ls_van
#        server_vanilla.grad_W=C_Gs_van
#        C_Ls_van=None
#        C_Gs_van=None
        gc.collect()
        #</FedSGD>
        omega_matrix = []
        for i in range(0, number_of_clients):
            omega_matrix.append(server.Alpha_matrix[:,i])
        loss_mat = []
        for i in range(0, number_of_clients):
            loss_mat.append(server.Losses[i])
        for i in range(0, number_of_clients):
            res=minimize(objective_fn,omega_matrix[i],loss_mat[i],bounds=bounds,constraints=constraint)
            server.Alpha_matrix[:,i]=res.x
        old_wts=np.squeeze(np.asarray(server.Weights))
        loops=0
        """Query new gradients"""
        print("Querying clients for gradients (Main Server)")
        [C_Gs, C_Ls]=hf.Query_Clients_Split(server,server.Weights,False, True,dataset_ID)#Single bit True or False. Query the gradients and losses at all clients
        server.grad_W=C_Gs
        gc.collect()
        w_gradient = hf.get_O_grad(server)
        learning_rate=rate
        if optimiser=='momentum':
            new_wts=hf.grad_des_asc(server.Weights,learning_rate,lr,-1)
        else:
            new_wts=hf.grad_des_asc(server.Weights,w_gradient,learning_rate,-1)
        w_gradient=None
        learning_rate=None
        gc.collect()
        temp=None
        #Converting the obtained weights to an appropriate format
        if server.number_of_clients==1:
            new_wts=[np.expand_dims(new_wts,axis=1)]
        else:
            new_wts=[np.expand_dims(temp,axis=1) for temp in new_wts]
        server.Weights=new_wts 
        temp_wts = new_wts
        new_wts=None
        print("Shared Gradient Descent Done")
        [C_Gs_t, C_Ls_t]=hf.Query_Clients_Split(server,server.Weights,False,False,dataset_ID)#Single bit True or False. Query the gradients and losses at all clients
        server.grad_W=C_Gs_t # contentious: just task specific gradient
#        server.grad_W=np.asarray(server.grad_W)+np.asarray(C_Gs_t) # contentious: addition of shared gradient with task specific gradient
        server.grad_W = [np.array(j) for j in server.grad_W]
        gc.collect()
        w_gradient_t = hf.get_O_grad(server)
        learning_rate_t=rate
        """ Gradient descent step"""
        if optimiser=='momentum':
            new_wts=hf.grad_des_asc(server.Weights,learning_rate_t,lr,-1)
        else:
            new_wts=hf.grad_des_asc(server.Weights,w_gradient_t,learning_rate_t,-1)
        """W-Gradient Descent Done"""
        w_gradient=None
        learning_rate=None
        #Converting the obtained weights to an appropriate format
        if server.number_of_clients==1:
            new_wts=[np.expand_dims(new_wts,axis=1)]
        else:
            new_wts=[np.expand_dims(temp,axis=1) for temp in new_wts]

        server.Weights = new_wts #New Query Weights
        server.finalise_client_weights(server.Weights) #Update learned weights to all clients
        temp=np.linalg.norm(old_wts-np.squeeze(new_wts),axis=1)
        weight_criterion=np.average(temp)
        weight_history.append(temp)
        temp=None
        gc.collect()
        print("Task-Specific Gradient Descent Done")
        """Save Client Weights to Drive """
        save_weights() #Function to save the weights of the clients onto disk
        new_wts=None
        new_wts_vanilla=None
        gc.collect()
        print("Gradient Descents Done")
        # calculating total loss
        [C_Gs_t, C_Ls_t]=hf.Query_Clients(server,server.Weights,False)#Single bit True or False. Query the gradients and losses at all clients
        server.Losses=C_Ls_t
        my_loss = []
        for i in range(0, number_of_clients):
            my_loss.append(np.dot(C_Ls_t[i],omega_matrix[i]))
#        print(my_loss)
        gc.collect()
        """ Change this"""
#        Loss_History.append(-1)
        Loss_History.append(my_loss)
        np.save(path_to_save_load+"Loss_History_"+programID,Loss_History)
        np.save(path_to_save_load+"weight_history_"+programID,weight_history)
        #Save as text file
        np.savetxt(text_save_path+"Loss_History_"+programID+".txt",Loss_History,fmt='%f',delimiter='\t')
        np.savetxt(text_save_path+"weight_history_"+programID+".txt",weight_history,fmt='%f',delimiter='\t')
        np.save(path_to_save_load+"weight_history_vanilla_"+programID,weight_history_vanilla)
        np.savetxt(text_save_path+"weight_history_vanilla_"+programID+".txt",weight_history_vanilla,fmt='%f',delimiter='\t')
        #Accuracies for MTFeeL
        [_,temp]=hf.Accuracy(server,server.cl)
        avg_acc.append(temp)
        np.save(path_to_save_load+"avg_acc_"+programID,avg_acc)
        np.savetxt(text_save_path+"avg_acc_"+programID+".txt",avg_acc,fmt='%f',delimiter='\t')
        
        #Accuracies for vanilla FL
        [z,temp]=hf.Accuracy(server_vanilla,server_vanilla.cl)
        avg_acc_vanilla.append(temp)
        np.save(path_to_save_load+"avg_acc_vanilla_"+programID,avg_acc_vanilla)
        np.savetxt(text_save_path+"avg_acc_vanilla_"+programID+".txt",avg_acc_vanilla,fmt='%f',delimiter='\t')
        
        """ ##########Global Epoch Updating##########"""
        all_local_epochs_done=True
        for clients in server.cl:
            #Check if all clients have completed at least one local epoch
            all_local_epochs_done=(all_local_epochs_done and clients.epoch_tracker)
        # print(all_local_epochs_done)
        if all_local_epochs_done:
            for clients in server.cl:
                #Allow all clients to shuffle data and reset local epochs count
                clients.allow_shuffle=True
            for clients_van in server_vanilla.cl:
                #Allow all clients to shuffle data and reset local epochs count
                clients_van.allow_shuffle=True
            global_epoch+=1 #One global epoch is completed
        iterations+=1
        if iterations%5==0: #Save Plots Every 5 Iterations
            hf.Graph_Me(programID,0,iterations)
        loops+=1
        print("\n")
    a=1/0 #Divide by zero to start plots of results # uncomment
except (KeyboardInterrupt,ZeroDivisionError) as e:
    #Save details of run
    details=["Train Samples ="+str(train_samples),
             "Test Samples ="+str(test_samples),
             "Batch size ="+str(server.cl[0].batch_size),
             "Number Of Clients "+str(number_of_clients),
             "Iterations "+str(iterations),
             "Regularizer Vanilla "+str(server_vanilla.gamma),
             "Regularizer MTFeeL"+str(server.gamma)
        ]
    np.savetxt(text_save_path+"Details_"+programID+".txt",details,fmt='%s')
    
    
    [acc_proposed_train,acc_proposed_test]=hf.Final_Results(server,weight_history,Loss_History,avg_acc)
    [acc_vanilla_train,acc_vanilla_test]=hf.Final_Results(server_vanilla,weight_history_vanilla,Loss_History_vanilla,avg_acc_vanilla)
    # np.save(path_to_save_load+"acc_proposed_train_"+programID,acc_proposed_train)
    np.savetxt(text_save_path+"acc_proposed_train_"+programID+".txt",acc_proposed_train,fmt='%f',delimiter='\t')
    np.savetxt(text_save_path+"acc_proposed_test_"+programID+".txt",acc_proposed_test,fmt='%f',delimiter='\t')
    np.savetxt(text_save_path+"acc_vanilla_train_"+programID+".txt",acc_vanilla_train,fmt='%f',delimiter='\t')
    np.savetxt(text_save_path+"acc_vanilla_test_"+programID+".txt",acc_vanilla_test,fmt='%f',delimiter='\t')
    print("".center(100,'#'))
    print("Summary".center(100,' '))
    print("".center(100,'#'))
    
    print("Locally Trained".center(100,'.'))
    # [print("Client {}--> {}% ".format(client,i*100)) for client,i in zip(range(server.number_of_clients),np.round(Accuracies_pretrained,3))]    

    print("Vanilla Federated".center(100,'*'))
    [print("Client {}--> {}% ".format(client,i*100)) for client,i in zip(range(server_vanilla.number_of_clients),np.round(acc_vanilla_test,3))] 
    print("Average Performance-->{:0.2f}%".format(100*np.average(acc_vanilla_test)))
    
    print("Proposed Algorithm".center(100,'.'))    
    [print("Client {}--> {}% ".format(client,i*100)) for client,i in zip(range(server.number_of_clients),np.round(acc_proposed_test,3))]    
    hf.finish()
    print("Average Performance-->{:0.2f}%".format(100*np.average(acc_proposed_test)))
    
#%% Final Tests
#for serv in [server]:
    # zzz=performances=hf.Confusion_Metric(serv,returns=True,use_train_set=True,verbose=False)
    #print(" Training Accuracy".center(100,' '))
    #print("".center(100,'_'))
    #np.savetxt(text_save_path+serv.name+"_train_accuracy_"+programID+".txt",zzz,fmt='%f',delimiter='\t')
    #[print(x*100) for x in zzz]
    
    # Testing Accuracy
    #zzz=performances=hf.Confusion_Metric(serv,returns=True,use_train_set=False,verbose=False)
    #np.savetxt(text_save_path+serv.name+"_test_accuracy_"+programID+".txt",zzz,fmt='%f',delimiter='\t')
    #print(" Testing Accuracy".center(100,' '))
    #print("".center(100,'_'))
    #[print(x*100) for x in zzz]