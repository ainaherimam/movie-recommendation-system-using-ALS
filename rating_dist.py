
from module import *
import matplotlib.pyplot as plt


def rating_plot_data(data):

    # Define a function 'rating_plot_data' to create data for plotting a ratings distribution.

    plot_data = {}

    for data_sample in data:
        for movie, rating in data_sample:
            
            if rating not in plot_data:
                plot_data[rating] = 1
            else:
                plot_data[rating] += 1

    # Return the keys (ratings) and values (counts)
    return plot_data.keys(), plot_data.values()


def plot_rating_dist(X, Y, save=''):

    # Define a function 'plot_rating_dist' to create a bar plot for the ratings distribution.

    plt.bar(X, Y, width=0.4)

    # Set X-axis ticks to show all the ratings
    plt.xticks(X)

    # Set labels and title for the plot.
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Ratings Distribution')


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


#Get plotting data for distribution
X_plot_rating,Y_plot_rating = rating_plot_data(data_by_user)


plot_rating_dist(list(X_plot_rating),list(Y_plot_rating),save='')