import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import different_plots as udplots


if __name__ == "__main__":

    graphs_save_location = "/Users/Rashid/Desktop/Visualization/graphs/scalability/"

    ps_results_4_cores = [92585, 55712, 31117]

    ring_results_4_cores = [91785, 53625, 27206]

    number_of_parallel_calls = [4]

    ps_title = 'Total completion time - Parameter server'

    save_as = "ps_npc"+str(number_of_parallel_calls[0])+".png"
    ylabel = 'Completion time(seconds)'
    xlabel = 'Number of workers'
    udplots.plot_bar(ps_results_4_cores, ps_title,xlabel,ylabel,graphs_save_location,save_as)

    ring_title = 'Total completion time - Ring AllReduce'

    save_as = "ring_npc" + str(number_of_parallel_calls[0]) + ".png"
    ylabel = 'Completion time(seconds)'
    xlabel = 'Number of workers'
    udplots.plot_bar(ring_results_4_cores, ring_title, xlabel, ylabel, graphs_save_location, save_as)

    psvsring = []
    psvsring.append(ps_results_4_cores)
    psvsring.append(ring_results_4_cores)

    save_as = "comparison_ttlcomptime_psvsring.png"
    plt_title = "Comparison of Total Completion Time"
    legends = ['Parameter Server', 'Ring AllReduce']
    legend_loc = "upper right"
    xlabel = 'Number of workers'
    ylabel = 'Completion Time (seconds)'

    udplots.grouped_bars(psvsring,graphs_save_location,xlabel,ylabel,save_as,plt_title,legends,legend_loc)

    # Local experiments.

    completion_time_local = [28599.52,18656.33,18205.51]
    save_as = "singlemachine_completiontime.png"
    plt_title = "Total Completion Time (Seconds)"
    xlabel = 'Number of GPUs'
    ylabel = 'Completion Time (seconds)'

    udplots.plot_bar(completion_time_local,plt_title,xlabel,ylabel,graphs_save_location,save_as)