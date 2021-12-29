import matplotlib.pyplot as plt


def plot(x, ys, legend_labels, title, out):
    for y in ys:
        plt.plot(x, y)
    # plt.ylim([0, 100])
    plt.xlabel = "epochs"
    plt.ylabel = title
    plt.legend(legend_labels)
    plt.title(title)
    plt.savefig(out)
    plt.clf()
