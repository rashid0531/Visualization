import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
if __name__ == "__main__":


    x = [0, 1, 2]
    y = [90, 40, 65]
    labels = ['high', 'low', 37337]
    plt.plot(x, y, 'r')
    plt.xticks(x, labels, rotation='vertical')
    plt.show()