import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import different_plots as udplots


def parse_lines_from_logs(file_name):

    with open(file_name,"r") as file_obj:

        lines = file_obj.readlines()

    filtered_lines= []


    for each_line in lines:

        # expected lines looks like : "utilization.gpu": 98,
        each_line = each_line.split(':')[-1]
        each_line = each_line.replace(',','')
        filtered_lines.append(int(each_line))

    return filtered_lines


def extract_utilization(which_experiment,input_path,graphs_save_location,number_of_workers):

    gpu_util_all_experiments = []

    for i in range(0,len(number_of_workers)):

        if ("Distributed" in which_experiment):

            if ("ps" in which_experiment):

                file_to_read = input_path + str(number_of_workers[i]) + "_workers/worker_1/util.txt"

            if ("ring" in which_experiment):

                file_to_read = input_path + str(number_of_workers[i]) + "_workers/util.txt"

            gpu_util_all_experiments.append(parse_lines_from_logs(file_to_read))


    avg_util = []
    variances = []
    stdeviation = []
    for i in range(0,len(gpu_util_all_experiments)):

        print(gpu_util_all_experiments[i])
        avg_util.append(np.mean(gpu_util_all_experiments[i]))
        variances.append(np.var(gpu_util_all_experiments[i]))
        stdeviation.append(np.std(gpu_util_all_experiments[i]))

    avg_util = list(map(lambda x: np.round(x,2),avg_util))
    variances = list(map(lambda x: np.round(x,2),variances))
    stdeviation = list(map(lambda x: np.round(x, 2), stdeviation))

    print(avg_util)
    print(stdeviation)
    print(variances)

    # x_pos = np.arange(len(gpu_util_all_experiments[0]))
    # plt.bar(x_pos, gpu_util_all_experiments[0], align='center', alpha=0.5)
    # plt.show()

if __name__ == "__main__":

    # input_path = "/Volumes/Samsung_T5/results/ring/results/ring_reduce_horovod/non_linear_learning_rate/4_threads/"
    # which_experiment = "Distributed_ring"

    input_path = "/Volumes/Samsung_T5/results/ps/thesis_results/4_threads/1_paramserver/"
    which_experiment = "Distributed_ps"

    graphs_save_location = "/Users/Rashid/Desktop/Visualization/graphs/scalability/utilization/"

    number_of_workers = ['2','4','8']

    extract_utilization(which_experiment,input_path,graphs_save_location,number_of_workers)

