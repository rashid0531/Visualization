import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import different_plots as udplots


def parse_lines_from_logs(experiment_name, file_name):

    with open(file_name,"r") as file_obj:

        lines = file_obj.readlines()

    filtered_lines= []

    if ("Distributed" in experiment_name):

        for each_line in lines:

            if ("INFO:tensorflow:Saving timeline for") not in each_line:

                if "loss" in each_line:

                    if (("INFO:tensorflow:global_step/sec:")) in each_line:

                        each_line = each_line.split(" ")
                        temp = ' '.join(each_line[:12])

                        filtered_lines.append(temp)

                    else:

                        filtered_lines.append(each_line)

        return filtered_lines


def experiment_dependant_parsing(experiment_name,lines):

    losses = []
    examples_per_seconds = []
    step_counts = []

    if ("Distributed" in experiment_name):

        for each_line in lines:

            examples_per_seconds.append(each_line.split(",")[-1].strip().split("=")[-1])

            losses.append(float(((each_line.split(",")[-2]).split("=")[-1].strip())))

            step_counts.append(each_line.split(",")[0].split(" ")[-1].strip())

        return losses,examples_per_seconds,step_counts


def get_accuracy_throughput(experiment_name,file_name):

    filtered_lines = parse_lines_from_logs(experiment_name,file_name)

    if filtered_lines == None:

        exit()

    losses,examples_per_seconds,step_counts = experiment_dependant_parsing(experiment_name,filtered_lines)

    print(step_counts[0],step_counts[1])

    indices = take_one_unique_step_from_repeated_steps(step_counts)

    losses = [losses[i] for i in indices]
    examples_per_seconds = [examples_per_seconds[i] for i in indices]

    # Filterd list of examples_per_seconds which doesn't contain empty values or "INFO:tensorflow:Saving timeline ...".
    examples_per_seconds = list(filter(lambda x: x != "", examples_per_seconds))
    examples_per_seconds = list(map(lambda x:x.strip().split(" ")[0], examples_per_seconds))

    # Convert Each element from string to float.
    examples_per_seconds = list(map(lambda x: float(x), examples_per_seconds))

    return losses,examples_per_seconds,step_counts

def take_one_unique_step_from_repeated_steps(array):

    """
    At the beginning of the training, some steps gets repeated several times due to the synchronization issues.
    This problem amortizes along time and stabilizes when the workers get synchronized.
    This functions finds the index of the last step among the repeated steps and merges with the index of the unique steps
    so that we can make a unique list of steps from a list which contains repeated and unique steps.

    :param array: array that contains the steps number as string
    :return: list of indexes of unique steps
    """
    indexes = []
    index_of_that_step = 0
    for i in range(1,len(array)):

        if int(array[index_of_that_step]) == int(array[i]):
            index_of_that_step = i
            #print(index_of_that_step)
        else:
            indexes.append(index_of_that_step)
            index_of_that_step = i

    # print(indexes[:5])
    # print("----------")
    return indexes


def visualize_experiments(which_experiment,input_path,batch_size_per_worker,total_number_of_images_in_dataset,number_of_epochs,number_of_workers,graphs_save_location):

    losses_per_epoch_for_all_experiments = []
    examples_per_seconds_for_all_experiments = []

    for i in range(0,len(number_of_workers)):

        if ("Distributed" in which_experiment):

            # file_to_read = input_path + str(number_of_workers[i]) +"_workers/worker_1/nohup.out"

            if ("ps" in which_experiment):

                file_to_read = input_path + str(number_of_workers[i]) + "_workers/worker_1/nohup.out"

            if ("ring" in which_experiment):

                file_to_read = input_path + str(number_of_workers[i]) + "_workers/nohup.out"

            losses, examples_per_seconds,step_indexes = get_accuracy_throughput(which_experiment,file_to_read)


        examples_per_seconds_for_all_experiments.append(examples_per_seconds)

        workers = number_of_workers[i]

        effective_batch_size = batch_size_per_worker * workers

        print(effective_batch_size)

        number_of_steps_in_each_epoch = int(np.floor(total_number_of_images_in_dataset / effective_batch_size))


        if ("Distributed" in which_experiment):

            loss_per_epoch = [losses[j] for j in range(0, len(losses)-number_of_steps_in_each_epoch, number_of_steps_in_each_epoch)]

            losses_per_epoch_for_all_experiments.append(loss_per_epoch)


    # Unfinished.
    min_value = np.min([len(losses_per_epoch_for_all_experiments[0]),len(losses_per_epoch_for_all_experiments[1]),len(losses_per_epoch_for_all_experiments[2])])

    for i in range(0,len(losses_per_epoch_for_all_experiments)):
        losses_per_epoch_for_all_experiments[i] = losses_per_epoch_for_all_experiments[i][:min_value]


    if ("ps" in which_experiment):

        if ("4" in which_experiment):

            plot_ttl = ('Box Plot : Throughput - Parameter server (npc = {})').format(str(4))

            plot_ttl = plot_ttl.replace("npc", "$\it{npc}$")

            file_name_to_save_as = "comparison_boxplot_4cores_ps.eps"

            bar_thrput_ttl = ('Average Throughput - Parameter server (npc = {})').format(4)

            bar_thrput_ttl = bar_thrput_ttl.replace("npc", "$\it{npc}$")

            bar_save_as = "thrput_bar_ps_npc" + str(4) + ".eps"

        if ("8" in which_experiment):

            plot_ttl = ('Box Plot : Throughput - Parameter server (npc = {})').format(str(8))

            plot_ttl = plot_ttl.replace("npc", "$\it{npc}$")

            file_name_to_save_as = "comparison_boxplot_8cores_ps.eps"

            bar_thrput_ttl = ('Average Throughput - Parameter server (npc = {})').format(8)

            bar_thrput_ttl = bar_thrput_ttl.replace("npc", "$\it{npc}$")

            bar_save_as = "thrput_bar_ps_npc" + str(8) + ".eps"


    if ("ring" in which_experiment):

        if ("4" in which_experiment):

            bar_thrput_ttl = 'Average Throughput - Ring AllReduce'

            bar_save_as = "thrput_bar_ring_npc" + str(4) + ".png"



        if ("8" in which_experiment):
            plot_ttl = ('Box Plot : Throughput - Ring all-reduce (npc = {})').format(str(8))

            plot_ttl = plot_ttl.replace("npc", "$\it{npc}$")

            file_name_to_save_as = "comparison_boxplot_8cores_ring.eps"


    # udplots.whisker_plot(examples_per_seconds_for_all_experiments,plot_ttl,graphs_save_location,file_name_to_save_as)


    avg_throughput = []
    variances = []
    stdeviation = []
    for i in range(0,len(examples_per_seconds_for_all_experiments)):

        if i == 2:
            print(np.sort(examples_per_seconds_for_all_experiments[i])[:23])

        avg_throughput.append(np.mean(examples_per_seconds_for_all_experiments[i]))
        variances.append(np.var(examples_per_seconds_for_all_experiments[i]))
        stdeviation.append(np.std(examples_per_seconds_for_all_experiments[i]))


    avg_throughput = list(map(lambda x: np.round(x,2),avg_throughput))
    variances = list(map(lambda x: np.round(x,2),variances))
    stdeviation = list(map(lambda x: np.round(x, 2), stdeviation))

    print(avg_throughput)
    print(stdeviation)
    print(variances)

    ylabel = 'Images/second'
    xlabel = 'Number of workers'
    #udplots.plot_bar(avg_throughput,bar_thrput_ttl,xlabel,ylabel,graphs_save_location,bar_save_as)



if __name__ == "__main__":

    """
    Each epoch is considered as an iteration of the whole dataset and each step represents processing one batch of images.
    So if in one epoch N number of images needs to be processed then the number of steps to complete one epoch = N / EB, where EB = effective batchsize = batch size per worker * number of workers.
    So we can set the "total number of steps" as "number of steps to complete one epoch" * "number of epochs".
    The loss values were calculated at each step during training. As the "total number of steps" can vary depending on the number of workers, but the "number of epochs" was set constant for each experiments, 
    it was decided to take a snapshot of the loss at the end of each epoch. 
    """

    graphs_save_location = "/Users/Rashid/Desktop/Visualization/graphs/scalability/throughput/"

    batch_size_per_worker = 8
    total_number_of_images_in_dataset = 1440
    number_of_epochs = 3000

    number_of_workers = [2,4,8]


    # input_path = "/Volumes/Samsung_T5/results/ring/results/ring_reduce_horovod/non_linear_learning_rate/4_threads/"
    #
    # which_experiment = "Distributed_ring_4cores"

    input_path = "/Volumes/Samsung_T5/results/ps/thesis_results/4_threads/1_paramserver/"
    which_experiment = "Distributed_ps_4cores"

    visualize_experiments(which_experiment,input_path,batch_size_per_worker,total_number_of_images_in_dataset,number_of_epochs,number_of_workers,graphs_save_location)

    # Average Thoughput

    ps_4_avg_thr = [48.41, 82.16, 156.2]
    ring_4_avg_thr = [47.89, 81.92, 162.49]

    ps_vs_ring_thrput = []
    ps_vs_ring_thrput.append(ps_4_avg_thr)
    ps_vs_ring_thrput.append(ring_4_avg_thr)

    save_as = "comparison_avgthrput_psvsring.png"
    plt_title = " Comparison of Average Throughput"
    legends = ['Parameter Server', 'Ring AllReduce']
    legend_loc = "upper left"
    xlabel = 'Number of workers'
    ylabel = 'Images/second'

    # udplots.grouped_bars(ps_vs_ring_thrput, graphs_save_location,xlabel,ylabel, save_as, plt_title, legends,legend_loc)