def plot_weight():
    import matplotlib.pyplot as plt

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    # Define the x-axis values (the labels from your data)
    x = ["$\leq$10", "$[$10, 20$]$", "$[$20, 30$]$", "$[$30, 40$]$", "${>}$ 40"]
    # Define the y-axis values (the weights from your data)
    BERT_Weight = [0.3659, 0.3988, 0.3727, 0.3621, 0.3707]
    Variation_Weight = [0.357625, 0.3659, 0.3942, 0.4034, 0.4051]
    Time_Weight = [0.2615, 0.2353, 0.2331, 0.2345, 0.2242]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylim([0.10, 0.7])
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # Plot the data
    ax.plot(x, BERT_Weight, label="BERT Weight", marker="o")
    ax.plot(x, Variation_Weight, label="Variation Weight", marker="o", color="red")
    ax.plot(x, Time_Weight, label="Time Weight", marker="o", color="purple")

    # Set the title and labels for the axes
    ax.set_title("")
    ax.set_xlabel(r"\textbf{Toys}", fontdict={"fontsize": 14})
    ax.set_ylabel("Weights")

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


def plot_weight2():
    import matplotlib.pyplot as plt

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    # Define the x-axis values (the labels from your data)
    x = ["$\leq$10", "$[$10, 20$]$", "$[$20, 30$]$", "$[$30, 40$]$", "${>}$ 40"]

    # Define the y-axis values (the weights from your data)
    BERT_Weight = [0.2239, 0.2183, 0.2156, 0.2179, 0.2203]
    Variation_Weight = [0.578525, 0.6342, 0.6342, 0.6434, 0.6412]
    Time_Weight = [0.197575, 0.1475, 0.1475, 0.1386, 0.1385]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.set_ylim([0.10, 0.7])

    # Plot the data
    ax.plot(x, BERT_Weight, label="BERT Weight", marker="o")
    ax.plot(x, Variation_Weight, label="Variation Weight", marker="o", color="red")
    ax.plot(x, Time_Weight, label="Time Weight", marker="o", color="purple")

    # Set the title and labels for the axes

    ax.set_xlabel(r"\textbf{Beauty}", fontdict={"fontsize": 14})
    ax.set_ylabel("Weights", fontdict={"fontsize": 14})

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


# plot_weight()
# plot_weight2()
