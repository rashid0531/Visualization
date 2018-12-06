import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(input_arr,method):

    """
    :param input_arr: an array of arrays each of which represents the losses of each experiments.
    For example, the first array contains the loss values of the experiment where 2 workers were used.
    Similar to that, the last array is for the experiment where 8 workers were used.
    :return:
    doesn't return anything. This function shows and saves the scatter plot of the losses of multiple experiments.

    """
    if (method == "normal"):
        # Get or set the current tick locations and labels of the x-axis.
        x_pos = np.arange(len(input_arr[0]))

        colors_and_patterns = ['r','g','b']
        for i in range(0,len(input_arr)):
        #plt.xticks(x_pos, input_arr)
            plt.plot(x_pos,input_arr[i],colors_and_patterns[i])
            plt.grid()
            plt.yscale('log')

    elif (method == "custom"):

        x_pos = [i*10 for i in range(0,len(input_arr[0]))]

        colors_and_patterns = ['r', 'g', 'b']
        for i in range(0, len(input_arr)):
            # plt.xticks(x_pos, input_arr)
            plt.plot(x_pos, input_arr[i], colors_and_patterns[i])
            plt.grid()
            plt.yscale('log')

    plt.show()

def whisker_plot(input_arr,plot_title,graphs_save_location,file_name_to_save_as):

    figure_value = 1
    fig = plt.figure(figure_value)
    fig.set_facecolor('white')

    plt.boxplot(input_arr)

    plot_title = plot_title
    plt.title(plot_title, size=11)

    # plt.yscale('log',basey=2, nonposy='clip')

    xpos = np.arange(len(input_arr)+1)
    print(xpos)
    xpos_labels = ["","2","4","8"]

    plt.xticks(xpos, xpos_labels)

    plt.ylabel('Images/second', size=9)
    plt.xlabel('Number of workers', size=9)

    plt.savefig(graphs_save_location + file_name_to_save_as, format='eps', dpi=1000)

    plt.show()


def plot_cdf(input_arr):

    figure_value = 2
    fig = plt.figure(figure_value)
    fig.set_facecolor('white')

    colors_and_patterns = ['r', 'g', 'b']

    plot_title = "Cumulative Distributed Function of throughput from different set of experiments (Parameter Server)"
    plt.title(plot_title, size=11)
    plt.xlabel('Throughput (images/second)', size=11)
    plt.ylabel('Cumulative Probability', size=11)

    xpos_labels = ["2 workers", "4 workers", "8 workers"]
    for i in range(0, len(input_arr)):

        sorted_data = np.sort(input_arr[i])
        plt.plot(sorted_data, np.linspace(0, 1, len(sorted_data)), color=colors_and_patterns[i],label=xpos_labels[i])

    plt.legend(loc="lower right")
    plt.show()

def plot_bar(input_arr,plot_title,xlabel,ylabel,graphs_save_location,file_name_to_save_as):

    xpos_labels = ["2", "4", "8"]
    x_pos = np.arange(len(xpos_labels))

    for a,b in zip(x_pos, input_arr):
        plt.text(a, b, str(b), ha = 'center', fontsize= 7)

    plt.bar(x_pos, input_arr, align='center',alpha=0.5)
    plt.xticks(x_pos, xpos_labels)
    plt.ylabel(ylabel,size=9)
    plt.xlabel(xlabel,size=9)
    plt.title(plot_title)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(graphs_save_location + file_name_to_save_as, format='eps', dpi=1000)

    plt.show()

def grouped_bars(arrays,graphs_save_location,xlabel,ylabel,save_as,plt_ttl,legends,legend_loc):

    Number_of_groups = 3

    # the x locations for the groups
    ind = np.arange(Number_of_groups)

    # the width of the bars
    width = 0.27

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(plt_ttl, size=13)
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    patterns = ["+", "x", "o", "O", ".", "*"]

    if ("Parameter Server vs Ring AllReduce" in plt_ttl):
        chosen_color = ['g','y']

    else :
        chosen_color = ['c','r']

    psvals = arrays[0]
    for a,b in zip(ind, psvals):
        plt.text(a, b, str(b), ha = 'center', fontsize= 7)

    rects1 = plt.bar(ind, psvals, width, color=chosen_color[0],edgecolor="black", hatch=patterns[1])

    ringvals = arrays[1]
    for a,b in zip(ind+ width, ringvals):
        plt.text(a, b, str(b), ha = 'center', fontsize= 7)
    rects2 = plt.bar(ind + width, ringvals, width, color=chosen_color[1],edgecolor="black", hatch=patterns[2])

    ax.set_ylabel(ylabel, size=11)

    # Positioning the xsticks.
    ax.set_xticks(ind+0.13)

    ax.set_xlabel(xlabel, size=11)
    ax.set_xticklabels(('2', '4', '8'))
    ax.legend((rects1[0], rects2[0]),
              (legends[0], legends[1]),
              loc=legend_loc, prop={'size': 15})

    plt.savefig(graphs_save_location + save_as,format='png', dpi=500)

    plt.show()

