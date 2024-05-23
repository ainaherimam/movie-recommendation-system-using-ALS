from module import *
import matplotlib.pyplot as plt

def data_for_plot_embedding(list_movie):
    selected_movies=[]

    # Loop through each row and select the first and second movies
    for i,row in enumerate(movie_vectors):
        if i in list_movie:
            selected_movies.append((row[0],row[1],movie_to_name[i])) 
        # First movie
    return selected_movies


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
number_of_users = len(data_by_user_train)
number_of_movie = len(data_by_movie_train)
latent_dimension = 2
nbr_iteration = 10
sigma = 1 / np.sqrt(latent_dimension)
lamda = 0.01
tau = 0.1
gamma = 0.05

# Initialize random latent vectors for movies and users
movie_vectors = np.random.normal(0, sigma, size=[number_of_movie, latent_dimension])
user_vectors = np.random.normal(0, sigma, size=[number_of_users, latent_dimension])
user_bias = np.zeros(number_of_users)
movie_bias = np.zeros(number_of_movie)

# Initialize lists to store training history
rmse_train = []
rmse_test = []
losses_train = []
losses_test = []
iteration = []


for i in range(nbr_iteration):

    # Update biases for users and movies.
    for user in range(number_of_users):
        update_bias(data_by_user_train, user_vectors, movie_vectors, movie_bias, user_bias, lamda, gamma, user)

    # Update latent vectors for users.
    for user in range(number_of_users):
        if not data_by_user_train[user]:
            continue
        update_vector(data_by_user_train, movie_vectors, user_vectors, user_bias, movie_bias, lamda, tau, user)

    # Update biases for movies.
    for movie in range(number_of_movie):
        update_bias(data_by_movie_train, movie_vectors, user_vectors, user_bias, movie_bias, lamda, gamma, movie)

    # Update latent vectors for movies.
    for movie in range(number_of_movie):
        if not data_by_movie_train[movie]:
            continue
        update_vector(data_by_movie_train, user_vectors, movie_vectors, movie_bias, user_bias, lamda, tau, movie)

    # Calculate the training and testing losses and RMSE.
    loss = loss_and_rmse(user_vectors, movie_vectors, data_by_user_train, data_by_movie_train, user_bias, movie_bias, lamda, tau, gamma)
    loss_test = loss_and_rmse(user_vectors, movie_vectors, data_by_user_test, data_by_movie_test, user_bias, movie_bias, lamda, tau, gamma)

    # Append the results to their respective lists.
    losses_train.append(loss[0])
    losses_test.append(loss_test[0])
    rmse_train.append(loss[1])
    rmse_test.append(loss_test[1])
    iteration.append(i)

    # Print iteration results.
    print(f"Iteration {i + 1}, rmse={loss[1]}  rmse_test={loss_test[1]}, loss={loss[0]}  loss_test={loss_test[0]}")





# Define a list of data for different movie genres
action_movie = data_for_plot_embedding([8, 55, 22, 210, 114, 962, 44, 550])
kids_movie = data_for_plot_embedding([66, 445, 256, 3, 224, 102, 12, 803])
horror_movie = data_for_plot_embedding([1520, 100, 1230, 170, 727, 272, 424, 44])
scifi_movie = data_for_plot_embedding([57, 988, 185, 851, 843, 2405, 1432, 639])
romance_movie = data_for_plot_embedding([22, 747, 52, 451, 52, 44, 89, 278])

# Create a list containing all the movie genre data
data = [action_movie, kids_movie, horror_movie, scifi_movie, romance_movie]

# Define markers and colors for each movie genre
markers = ['h', '^', 'D', 'v', '*']
colors = ['g', 'r', 'c', 'y', 'b']
labels = ['Action', 'Kids', 'Horror', 'Sci-fi', 'Romance']

plt.figure(figsize=(8, 8))
ax = plt.gca()

# Create a scatter plot for each movie genre
for i, sublist in enumerate(data):
    x = [point[0] for point in sublist]
    y = [point[1] for point in sublist]
    marker = markers[i]
    color = colors[i]
    label = labels[i]
    plt.scatter(x, y, marker=marker, color=color, label=label)


ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='both', length=0)

# Set the limits for the plot
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.axhline(0, color='gray', linewidth=2)
plt.axvline(0, color='gray', linewidth=2)
plt.title('')
plt.legend()

# Show the plot
plt.show()
