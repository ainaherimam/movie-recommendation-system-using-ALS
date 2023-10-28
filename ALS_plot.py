from module import *

dataset_path = 'data/ratings.csv'
movie_info_path = 'data/movies.csv'

#Load the data
dataset = load_data(dataset_path)
movie_information = load_movie_info(movie_info_path)

#Create the data structure for sparse matrix
system_user_to_user_dict,user_to_system_user,system_mov_to_mov_dict,mov_to_system_mov,data_by_user_train,data_by_movie_train,data_by_user_test,data_by_movie_test=structure_split_data(dataset)

#create list to map index to movie title
movie_to_name = [get_movie_info(i,mov_to_system_mov,movie_information)[0] for i in range(len(mov_to_system_mov))]

#create list to map index to movie features
movie_to_feature = [get_movie_info(i,mov_to_system_mov,movie_information)[1] for i in range(len(mov_to_system_mov))]



#Train the model with ALS
#Initialize hyperparameters 
number_of_users=len(data_by_user_train)
number_of_movie=len(data_by_movie_train)
latent_dimension=5
nbr_iterration=10
sigma= 1/np.sqrt(latent_dimension)
lamda= 0.01
tau= 0.1
gamma= 0.05

movie_vectors = np.random.normal(0,sigma,size=[number_of_movie,latent_dimension])
user_vectors = np.random.normal(0,sigma,size=[number_of_users,latent_dimension])
user_bias=np.zeros(number_of_users)
movie_bias=np.zeros(number_of_movie)

rmse_train=[]
rmse_test=[]
losses_train=[]
losses_test=[]
iteration=[]

for i in range(nbr_iterration):

  for user in range(number_of_users):
    update_bias(data_by_user_train,user_vectors,movie_vectors,movie_bias,user_bias,lamda,gamma,user)

  for user in range(number_of_users):
    if not data_by_user_train[user]:
      continue
    update_vector(data_by_user_train,movie_vectors,user_vectors,user_bias, movie_bias,lamda,tau,user)

  for movie in range(number_of_movie):
    update_bias(data_by_movie_train,movie_vectors,user_vectors,user_bias,movie_bias,lamda,gamma,movie)

  for movie in range(number_of_movie):
    if  not data_by_movie_train[movie]:
      continue
    update_vector(data_by_movie_train,user_vectors,movie_vectors,movie_bias,user_bias,lamda,tau,movie)

  loss=loss_and_rmse(user_vectors,movie_vectors,data_by_user_train,data_by_movie_train, user_bias,movie_bias,lamda,tau,gamma)
  loss_test=loss_and_rmse(user_vectors,movie_vectors,data_by_user_test,data_by_movie_test, user_bias,movie_bias,lamda,tau,gamma)

  losses_train.append(loss[0])
  losses_test.append(loss_test[0])
  rmse_train.append(loss[1])
  rmse_test.append(loss_test[1])
  iteration.append(i)

  print(f"It {i+1}, rmse={loss[1]}  rmse_test={loss_test[1]}, loss={loss[0]}  loss_test={loss_test[0]}")


# print(movie_to_name)
plot_log_and_rmse(X=iteration, Y=rmse_train, Z=rmse_test, Y2=losses_train, Z2=losses_test)
