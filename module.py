import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv
import pickle


def load_data(dataset_path):

    # Create an empty list to store the dataset.
    dataset = []

    with open(dataset_path, 'r') as data_file:

        # Create a CSV reader object to parse the contents of the file
        data_reader = csv.reader(data_file, delimiter=',')

        # Skip the first row  (column headers).
        next(data_reader)

        # Iterate through each row in the CSV file.
        for row in data_reader:

            user_system_id, movie_system_id, rating, timestamp = row

            # remove timestamp and add to dataset
            dataset.append([user_system_id, movie_system_id, rating])

    return np.array(dataset)





def load_movie_info(movie_info_path):
    
    movie_info = {}

    # Open the file specified by 'movie_info_path'
    with open(movie_info_path, 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',')

        # Skip the first row of the CSV file (often used for column headers).
        next(data_reader)

        # Iterate through each row in the CSV file 
        for row in data_reader:
            # Extract movie information from the row.
            movie_id = row[0]
            movie_name = row[1]
            movie_feature_list = row[2]

            movie_info[movie_id] = (movie_name, movie_feature_list)

    return movie_info

def structure_data(arr):

  #intialize all the data structure
  system_user_to_user_dict =  {}
  user_to_system_user = []
  system_mov_to_mov_dict = {}
  mov_to_system_mov = []
  data_by_user = []
  data_by_movie= []

  #create all the data structure using in a loop
  for user_sys, movie_sys, rating in arr:

      #Take care of the user data structure
      if user_sys not in system_user_to_user_dict:
        user_to_system_user.append(user_sys)
        system_user_to_user_dict[user_sys] = len(system_user_to_user_dict)
        data_by_user.append([])

      #Take care of the movie data structure
      if movie_sys not in system_mov_to_mov_dict:
        mov_to_system_mov.append(movie_sys)
        system_mov_to_mov_dict[movie_sys] = len(system_mov_to_mov_dict)
        data_by_movie.append([])

      #Simulate the sparse matrix with a list of list of tuplesdot_products = np.sum([np.dot(v, v) for v in v_n])


      data_by_user[system_user_to_user_dict[user_sys]].append((system_mov_to_mov_dict[movie_sys],float(rating)))
      data_by_movie[system_mov_to_mov_dict[movie_sys]].append((system_user_to_user_dict[user_sys],float(rating)))

  return system_user_to_user_dict,user_to_system_user,system_mov_to_mov_dict,mov_to_system_mov,data_by_user,data_by_movie

def structure_split_data(arr):
  #Shuffle the data

  split_point = 0.9 * len(arr)

  #intialize all the data structure
  system_user_to_user_dict = {}
  user_to_system_user = []
  system_mov_to_mov_dict = {}
  mov_to_system_mov = []

  # First build the mappings.
  for index in range(len(arr)):

    user_sys = arr[index][0]
    movie_sys = arr[index][1]

    #Take care of the user data structure
    if user_sys not in system_user_to_user_dict:
      user_to_system_user.append(user_sys)


      system_user_to_user_dict[user_sys] = len(system_user_to_user_dict)
  
    #Take care of the movie data structure
    if movie_sys not in system_mov_to_mov_dict:
      mov_to_system_mov.append(movie_sys)
      system_mov_to_mov_dict[movie_sys]=len(system_mov_to_mov_dict)
  
  #Initialize with empty list all the *trainings* data.
  data_by_user_train = [[] for i in range(len(user_to_system_user))]
  data_by_movie_train = [[] for i in range(len(mov_to_system_mov))]

  #Initialize with empty list all the *test* data.
  data_by_user_test = [[] for i in range(len(user_to_system_user))]
  data_by_movie_test = [[] for i in range(len(mov_to_system_mov))]


  #create all the data structmov_to_system_movure using in a loop
  for index in range(len(arr)):

    user_sys = arr[index][0]
    movie_sys = arr[index][1]
    rating = arr[index][2]

    user_index = system_user_to_user_dict[user_sys]
    movie_index = system_mov_to_mov_dict[movie_sys]

    if index < split_point:
      # Insert into the sparse user and item *training* matrices.
      data_by_user_train[user_index].append((movie_index, float(rating)))
      data_by_movie_train[movie_index].append((user_index, float(rating)))

    else:
      # Insert into the sparse user and item *test* matrices.
      data_by_user_test[user_index].append((movie_index, float(rating)))
      data_by_movie_test[movie_index].append((user_index,float(rating)))


  
  return system_user_to_user_dict,user_to_system_user,system_mov_to_mov_dict,mov_to_system_mov,data_by_user_train,data_by_movie_train,data_by_user_test,data_by_movie_test




def get_movie_info(movie_index,mov_to_system_mov,movie_information):
    # Function created to get information from a index and return a tuple (movie_name,movie_features)
    sys_movie = mov_to_system_mov[movie_index]

    return movie_information[sys_movie]





def search_movies(movie_list, search_string):
  # Function used to search for movies in the database
    for index,movie_name in enumerate(movie_list):
        if search_string in movie_name:
            print(f'sparse index:{index} , name: {movie_name}, feature: {movie_to_feature_orig[index]}')




def write(file,vector):
  #Write down a file with pickle
    with open(file, 'wb') as file:
        pickle.dump(vector, file)




def load(current_directory,filename):
  #Load up a file with pickle
    with open(current_directory+'/'+filename, 'rb') as file:

        #Create a variable correponding to the vector file
        globals()[filename[:-4]] = pickle.load(file)


def loss_and_rmse(user_vector,movie_vector,data_by_user,data_by_movie,user_bias, movie_bias, lamda,tau,gamma):
  
  #Function for computing the loss and rmse

  loss = 0
  #Rating counter
  nbr_rating = 0
  for i in range(len(data_by_user)):
    for movie , rating in data_by_user[i]:
      loss += (rating - user_vector[i].T @ movie_vector[movie] - user_bias[i] - movie_bias[movie]) **2
      nbr_rating += 1


  #Calculate the user regularizer
  user_reg = 0
  for m in range(len(data_by_user)):
    user_reg += user_vector[m].T @ user_vector[m]

  #Calculate the movie regularizer
  movie_reg = 0
  for n in range(len(data_by_movie)):
    movie_reg += movie_vector[n].T @ movie_vector[n]

  #Calculate the user bias regularizer
  user_bias_sum = np.dot(user_bias,user_bias)


   #Calculate the movie bias regularizer
  movie_bias_sum = np.dot(movie_bias,movie_bias)

  #The loglikelihood
  loglikelihood = -1*(lamda/2)*loss-1*(tau/2)*user_reg+-1*(tau/2)*movie_reg -1*(gamma/2)*user_bias_sum - 1*(gamma/2)*movie_bias_sum
  
  #RMSE
  rmse=np.sqrt(loss/nbr_rating)

  return -loglikelihood,rmse



def rmse(user_vector,movie_vector,data_by_user,user_bias, movie_bias):

  #Function for computing the root mean squared error
  
  loss=0
  nbr_rating = 0

  for i in range(len(data_by_user)):
    for movie,rating in data_by_user[i]:

      loss+=(rating - user_vector[i].T @ movie_vector[movie] - user_bias[i] - movie_bias[movie])**2

      #Count the number of ratings
      nbr_rating += 1

  #Calculate the rmse
  rmse=np.sqrt(loss/nbr_rating)
  return rmse




def update_vector(data, vector, vector_to_update, bias_1, bias_2, lamda, tau, m):
    # Determine the number of latent dimension
    latent_dimension = vector.shape[1]


    matrix_left = np.zeros((latent_dimension, latent_dimension))
    matrix_right = np.zeros((latent_dimension,))

    # Iterate through the ratings associated with the user or movie 'm'.
    for index, rating in data[m]:

        Vn = vector[index]
        matrix_left += np.outer(Vn, Vn)

        matrix_right += Vn * (rating - bias_1[m] - bias_2[index])

    # Apply regularization to the left matrix and calculate its inverse.
    matrix_left = lamda * matrix_left + tau * np.identity(latent_dimension)
    matrix_left = np.linalg.inv(matrix_left)

    # Adjust the right matrix by applying regularization.
    matrix_right = lamda * matrix_right

    # Calculate the updated latent vector for user or movie 'm'.
    result = matrix_left @ matrix_right

    # Update the 'vector_to_update' with the new latent vector.
    vector_to_update[m] = result

    return result




def update_bias(data, vector_1, vector_2, bias, bias_to_change, lamda, gamma, m):


    bias_sum = 0

    # Iterate through the ratings associated with the user or movie 'm'.
    for index, rating in data[m]:
        # Calculate the sum of biases for this user or movie.
        bias_sum += (rating - vector_1[m].T @ vector_2[index] - bias[index])

 
    result = (lamda * bias_sum) / (lamda * len(data[m]) + gamma)

    # Update the bias_to_change array with the new bias value.
    bias_to_change[m] = result

    # Return the updated bias term for user or movie 'm'.
    return result


def plot_log_and_rmse(X, Y, Z, Y2, Z2, label_Y='train dataset', label_Z = 'test dataset', 
                      color_Y = 'red', color_Z='blue', axis_x = 'Iteration', axis_y1 = 'RMSE', 
                      title='RMSE', label_Y2 ='train dataset', label_Z2 ='test dataset', 
                      axis_y2='loss', color_Y2 ='red', color_Z2 = 'blue',
                      title2 = 'Negative log-likelihood', save=''):

    figsize = (11,6)  # Adjust the figsize as needed
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    #Take care of the first figure
    ax[0].scatter(X, Y, color='black', marker='.', s=40, label='')
    ax[0].scatter(X, Z, color='black', marker='.', s=40, label='')
    ax[0].plot(X, Y, color_Y, linestyle='-', linewidth=1, label=label_Y)
    ax[0].plot(X, Z, color_Z, linestyle='-', linewidth=1, label=label_Z)
    ax[0].set_xlabel(axis_x)
    ax[0].set_ylabel(axis_y1)
    ax[0].set_title(title)
    ax[0].legend()

    #Take care of the second figure
    ax[1].scatter(X, Y2, color='black', marker='.', s=40, label='')
    ax[1].scatter(X, Z2, color='black', marker='.', s=40, label='')
    ax[1].plot(X, Y2, color_Y2, linestyle='-', linewidth=1, label=label_Y2)
    ax[1].plot(X, Z2, color_Z2, linestyle='-', linewidth=1, label=label_Z2)
    ax[1].set_xlabel(axis_x)
    ax[1].set_ylabel(axis_y2)  # You can adjust this if needed
    ax[1].set_title(title2)  # You can change the title if needed
    ax[1].legend()

    if not save == '':
        plt.savefig(save)

    plt.show()


def predict(movie_index,movie_vectors,movie_bias):
    """
    Returns a list of tuple (movie_index,score) of top 10 recommendations for the choosen movie
    
    """
    
    #Create a dummy user
    dummy_user_vector= movie_vectors[movie_index].copy()
    
    #Get the prediction of all movie vectors
    recomender = [(i,(dummy_user_vector @ movie_vectors[i]) + 0.05*movie_bias[i]) for i in range(len(movie_vectors))]

    #Sort the recomender from the worse to the best
    recomender = sorted(recomender, key=lambda x: x[1],reverse=True)

    #Delete the given movie from the recomenation list
    recomender= [item for item in recomender if movie_index not in item]

    #returm the 10 best recommendations
    return recomender[:10]