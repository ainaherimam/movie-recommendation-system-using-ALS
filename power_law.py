from module import *
import matplotlib.pyplot as plt




def plotting_power_law(x1, x2, y1, y2, figsize=(6, 4), save=''):
    # Function for creating a power law plot.
    fig, ax = plt.subplots(figsize=figsize)

    # scatter plots 
    ax.scatter(x2, y2, label='data_by_movie')
    ax.scatter(x1, y1, color='red', marker='*', label='data_by_user')

    # Set the y-scale and x-scale to logarithmic.
    plt.yscale("log")
    plt.xscale("log")


    ax.set_xlabel('$Degree$')
    ax.set_ylabel('$Frequency$')
    ax.set_title('Power law')


    ax.legend()

 
    if save != '':
        plt.savefig(save)


    plt.show()

dataset_path = 'data/ratings.csv'
movie_info_path = 'data/movies.csv'

#Load the data
dataset = load_data(dataset_path)
movie_information = load_movie_info(movie_info_path)

#Create the data structure for sparse matrix
system_user_to_user_dict,user_to_system_user,system_mov_to_mov_dict,mov_to_system_mov,data_by_user,data_by_movie=structure_data(dataset)


#Get plotting data for the power law
user_plot_data_X = [len(i) for i in data_by_user]
movie_plot_data_X = [len(i) for i in data_by_movie]
user_plot_data_Y = [user_plot_data_X.count(i) for i in user_plot_data_X]
movie_plot_data_Y = [movie_plot_data_X.count(i) for i in movie_plot_data_X]


plotting_power_law(user_plot_data_X,movie_plot_data_X,user_plot_data_Y,movie_plot_data_Y)
