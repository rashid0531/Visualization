import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def parse_lines_from_logs(experiment_name, file_name):

    with open(file_name,"r") as file_obj:

        lines = file_obj.readlines()

    filtered_lines= []

    if experiment_name == "Single_Machine":

        # A sample line from the log file
        # 2018-08-26 23:04:42.294398: step 0, loss = 132.64 (23.1 examples/sec; 1.386 sec/batch)

        for each_line in lines:

            if ("loss" in each_line) and ("Final" not in each_line):

                each_line = each_line.replace("(", ", ").replace(")", "").replace(";", ",")

                filtered_lines.append(each_line)

        return filtered_lines


def experiment_dependant_parsing(experiment_name,lines):

    losses = []
    examples_per_seconds = []
    step_counts = []

    if experiment_name == "Single_Machine":

        #print(lines[13498])

        for id,each_line in enumerate(lines):

            examples_per_seconds.append(each_line.split(",")[-2].strip().split(" ")[0])

            losses.append(float(((each_line.split(",")[1]).split(" ")[-2])))

            step_counts.append(each_line.split(",")[0].split(" ")[-1].strip())

        return losses, examples_per_seconds, step_counts


def get_accuracy_throughput(experiment_name,file_name):

    filtered_lines = parse_lines_from_logs(experiment_name,file_name)

    if filtered_lines == None:

        exit()

    losses,examples_per_seconds,step_counts = experiment_dependant_parsing(experiment_name,filtered_lines)

    examples_per_seconds = list(filter(lambda x: x != "", examples_per_seconds))
    examples_per_seconds = list(map(lambda x:x.strip().split(" ")[0], examples_per_seconds))

    # Convert Each element from string to float.
    examples_per_seconds = list(map(lambda x: float(x), examples_per_seconds))

    return losses,examples_per_seconds,step_counts


def find_the_appropiate_epoch_indexes(losses,number_of_steps_in_each_epoch,step_indexes):

    list_of_tuples = []
    for i in range(0,len(step_indexes)):

        if (int(step_indexes[i]) % int(number_of_steps_in_each_epoch) == 0):

            # Saving a tuple of epoch_index,loss_value as a key value pair.
            key_value_tuple = ((int(step_indexes[i]) / int(number_of_steps_in_each_epoch)),losses[i])

            list_of_tuples.append(key_value_tuple)

    return list_of_tuples

def visualize_experiments(which_experiment,input_path,batch_size_per_worker,total_number_of_images_in_dataset,number_of_epochs,number_of_workers):

    losses_per_epoch_for_all_experiments = []
    examples_per_seconds_for_all_experiments = []

    for i in range(0,len(number_of_workers)):

        if (which_experiment == "Single_Machine"):

            # Unfinished.........
            file_to_read = input_path + str(number_of_workers[i]) + "_GPU/nohup.out"
            losses, examples_per_seconds,step_indexes = get_accuracy_throughput(which_experiment,file_to_read)


        examples_per_seconds_for_all_experiments.append(examples_per_seconds)

        workers = number_of_workers[i]

        effective_batch_size = batch_size_per_worker * workers

        number_of_steps_in_each_epoch = int(np.floor(total_number_of_images_in_dataset / effective_batch_size))

        # print("steps : ", number_of_steps_in_each_epoch)

        if (which_experiment == "Single_Machine"):

            """
            Things got complicated when the log frequency was set to 10 steps during training. This effects the way
            we calculated loss after each epoch is finished. As right now, the experiments conducted on Octopus is not possible to be rerun,
            a partial solution is conducted to mitigate the problem, by finding the last steps of each epoch for each single machine experiments.
            In order to match the epoch indexes, only common epoch indexes among all set of single machine experiments have been logged.    
            """

            loss_per_epoch = find_the_appropiate_epoch_indexes(losses,number_of_steps_in_each_epoch,step_indexes)

            # print(loss_per_epoch)

            """
            For single machine experiment, the "loss_per_epoch" contains a list of key, value dictionaries where epoch index is saved as key
            and loss as value. Keeping the key value pairs will help to find the common epoch indexes for all experiments so that plotting the loss functions
            looks reasonable.
            """

            losses_per_epoch_for_all_experiments.append(loss_per_epoch)


    epoch_idx = [[] for i in range(0,3)]

    for i in range(0,len(losses_per_epoch_for_all_experiments)):
        for j in range(0,len(losses_per_epoch_for_all_experiments[i])):
            epoch_idx[i].append(losses_per_epoch_for_all_experiments[i][j][0])


    common_epoch_idx = set(epoch_idx[0]) & set(epoch_idx[1]) & set(epoch_idx[2])

    common_epoch_idx = [int(each_element) for each_element in common_epoch_idx]

    common_epoch_idx = np.sort(common_epoch_idx)
    # print(common_epoch_idx)
    # print(len(common_epoch_idx))

    common_loss_values = [[] for i in range(0, 3)]
    for i in range(0, len(losses_per_epoch_for_all_experiments)):
        for j in range(0, len(losses_per_epoch_for_all_experiments[i])):

            if (int(losses_per_epoch_for_all_experiments[i][j][0]) in common_epoch_idx):

                common_loss_values[i].append(losses_per_epoch_for_all_experiments[i][j][1])


    # Take loss values upto 6000 epochs.
    for i in range(0, len(common_loss_values)):
        common_loss_values[i] = common_loss_values[i][1:601]

    common_epoch_idx = common_epoch_idx[1:601]

    colors_and_patterns = ['r', 'g', 'b']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Loss during training", size=13)
    fig.set_facecolor('white')

    labels = ['2 GPUS', '4 GPUS', '8 GPUS']
    for i in range(0, len(common_loss_values)):

        print(len(common_loss_values[i]))

        x_pos = np.arange(len(common_epoch_idx))
        ax.plot(x_pos, common_loss_values[i], colors_and_patterns[i],label=labels[i])
        # plt.xticks(x_pos,idx_values)
        ax.set_xticklabels(['','0','1000', '2000', '3000', '4000','5000','6000'])
        # plt.xticks(ax.get_xticks(),('','0','1000', '2000', '3000', '4000','5000','6000',''))

        plt.yscale('log')
        plt.ylabel("loss", size=11)
        plt.xlabel("number of epochs", size=11)
        ax.legend(loc='upper right', frameon=True)

    plt.savefig("loss_single_machine.png", format='png', dpi=500)
    plt.show()

if __name__ == "__main__":

    """
    Each epoch is considered as an iteration of the whole dataset and each step represents processing one batch of images.
    So if in one epoch N number of images needs to be processed then the number of steps to complete one epoch = N / EB, where EB = effective batchsize = batch size per worker * number of workers.
    So we can set the "total number of steps" as "number of steps to complete one epoch" * "number of epochs".
    The loss values were calculated at each step during training. As the "total number of steps" can vary depending on the number of workers, but the "number of epochs" was set constant for each experiments, 
    it was decided to take a snapshot of the loss at the end of each epoch. 
    """

    batch_size_per_worker = 32
    total_number_of_images_in_dataset = 1440
    number_of_epochs = 3000

    number_of_workers = [2,4,8]

    input_path = "/Volumes/Samsung_T5/results/accuracy/single_machine/xavier/train/"
    which_experiment = "Single_Machine"

    visualize_experiments(which_experiment,input_path,32,total_number_of_images_in_dataset,6000,number_of_workers)



