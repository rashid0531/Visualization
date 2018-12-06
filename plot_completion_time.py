import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import different_plots as udplots


if __name__ == "__main__":

    graphs_save_location = "/Users/Rashid/Desktop/Visualization/graphs/scalability/"

    batch_size_per_worker = 8
    total_number_of_images_in_dataset = 1440
    number_of_epochs = 3000

    number_of_workers = [2, 4, 8]

    # input_path = "/Users/Rashid/Desktop/thesis_results/Distributed/1_paramserver/"
    input_path = "/Users/Rashid/Desktop/thesis_results/Horovod/"
    which_experiment = "Distributed"

    ps_reuslts_4_cores = [92585, 55712, 31117]
    ps_reuslts_8_cores = [102773, 80884, 73848]

    ring_results_4_cores = [2500, 5000, 7500]
    ring_results_8_cores = [2500, 5000, 7500]

    number_of_parallel_calls = [4,8]

    ps_title = ('Total completion time - Parameter server (npc = {})').format(str(number_of_parallel_calls[0]))
    ps_title = ps_title.replace("npc","$\it{npc}$")
    save_as = "ps_npc"+str(number_of_parallel_calls[0])+".eps"
    ylabel = 'Completion time(seconds)'
    xlabel = 'Number of workers'
    udplots.plot_bar(ps_reuslts_4_cores, ps_title,xlabel,ylabel,graphs_save_location,save_as)

    ps_title = ('Total completion time - Parameter server (npc = {})').format(str(number_of_parallel_calls[1]))
    ps_title = ps_title.replace("npc", "$\it{npc}$")
    save_as = "ps_npc" + str(number_of_parallel_calls[1])+".eps"
    ylabel = 'Completion time(seconds)'
    xlabel = 'Number of workers'
    udplots.plot_bar(ps_reuslts_8_cores, ps_title,xlabel,ylabel,graphs_save_location,save_as)

    ps_4vs8cores = []
    ps_4vs8cores.append(ps_reuslts_8_cores)
    ps_4vs8cores.append(ps_reuslts_4_cores)

    save_as = "comparison_ps_ttlcomptime_4vs8.png"
    plt_title = " Total Completion Time - Parameter Server"
    legends = ['$\it{npc}$ = 8', '$\it{npc}$ = 4']
    legend_loc = "upper right"
    xlabel = 'Number of workers'
    ylabel = 'Completion Time (seconds)'

    udplots.grouped_bars(ps_4vs8cores,graphs_save_location,xlabel,ylabel,save_as,plt_title,legends,legend_loc)


    # Results from Horovod.

    # ring_title = ('Total completion time - Ring AllReduce (npc = {})').format(str(number_of_parallel_calls[0]))
    # ring_title = ring_title.replace("npc","$\it{npc}$")
    # save_as = "ring_npc"+str(number_of_parallel_calls[0])+".eps"
    # ylabel = 'Completion time(seconds)'
    # xlabel = 'Number of workers'
    # udplots.plot_bar(ring_results_4_cores, ring_title,xlabel,ylabel,graphs_save_location,save_as)
    #
    # ring_title = ('Total completion time - Ring AllReduce (npc = {})').format(str(number_of_parallel_calls[1]))
    # ring_title = ring_title.replace("npc", "$\it{npc}$")
    # save_as = "ring_npc" + str(number_of_parallel_calls[1])+".eps"
    # ylabel = 'Completion time(seconds)'
    # xlabel = 'Number of workers'
    # udplots.plot_bar(ring_results_8_cores, ring_title,xlabel,ylabel,graphs_save_location,save_as)
    #
    # ring_4vs8cores = []
    # ring_4vs8cores.append(ring_results_4_cores)
    # ring_4vs8cores.append(ring_results_8_cores)
    #
    # save_as = "comparison_ring_4vs8.png"
    # plt_title = "Impact of different values of $\it{npc}$ in input pipeline (Ring AllReduce)"
    # legends = ['$\it{npc}$ = 4', '$\it{npc}$ = 8']
    # udplots.grouped_bars(ring_4vs8cores,graphs_save_location,save_as,plt_title,legends)
    #
    # # Horovod vs Distributed Tensorflow.
    #
    # psvsring_4cores = []
    # psvsring_4cores.append(ps_reuslts_4_cores)
    # psvsring_4cores.append(ring_results_4_cores)
    #
    # save_as = "comparison_psvsring_4.png"
    # plt_title = " Total Completion Time - Parameter Server vs Ring AllReduce ($\it{npc}$ = 4)"
    # legends = ['Parameter Server', 'Ring AllReduce']
    # udplots.grouped_bars(psvsring_4cores, graphs_save_location, save_as, plt_title, legends)
    #
    #
    # psvsring_8cores = []
    # psvsring_8cores.append(ps_reuslts_8_cores)
    # psvsring_8cores.append(ring_results_8_cores)
    #
    # save_as = "comparison_psvsring_8.png"
    # plt_title = " Total Completion Time - Parameter Server vs Ring AllReduce ($\it{npc}$ = 8)"
    # legends = ['Parameter Server', 'Ring AllReduce']
    # udplots.grouped_bars(psvsring_8cores, graphs_save_location, save_as, plt_title, legends)