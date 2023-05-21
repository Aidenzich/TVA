# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from typing import Any


def plot_scores(
    param_name: str,
    metric_name: str,
    param_values: list,
    beauty: list,
    toys: list,
    ml_1m: list,
    ax1_ylim_min: float = 0.16,
    ax1_ylim_max: float = 0.18,
    ax2_ylim_min: float = 0.04,
    ax2_ylim_max: float = 0.06,
) -> Any:
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    # If we were to simply plot the data, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 4))
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    # param_values = [2 * i for i in param_values]

    # plot the same data on both axes
    lns1 = ax2.plot(param_values, beauty, marker="o")
    lns2 = ax2.plot(param_values, toys, marker="o")

    lns3 = ax1.plot(param_values, ml_1m, marker="o", color="r")

    lns = lns1 + lns2 + lns3
    labs = ["Beauty", "Toys", "ML-1M"]
    ax1.legend(lns, labs, loc="upper left")

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(None, ax1_ylim_max)  # outliers only
    ax2.set_ylim(ax2_ylim_min, ax2_ylim_max)  # most of the data

    # Set the interval of the y-axis to 0.005
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.005))

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top

    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=6,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.xlabel(r"\textbf{" + param_name + "}", fontdict={"fontsize": 14})
    plt.ylabel(r"\textbf{" + metric_name + "}", fontdict={"fontsize": 14}, loc="top")

    return plt


def exp_length() -> None:
    param_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    beauty = [
        0.05284,
        0.05584,
        0.05448,
        0.05542,
        0.05455,
        0.05569,
        0.05408,
        0.05274,
        0.05104,
    ]
    toys = [
        0.04913,
        0.05148,
        0.05393,
        0.05289,
        0.05282,
        0.05366,
        0.05199,
        0.05325,
        0.05163,
    ]
    ml_1m = [0.1652, 0.1729, 0.1753, 0.1703, 0.1772, 0.1721, 0.1781, 0.1704, 0.1687]
    plot_scores("(a)Length", "NDCG@10", param_values, beauty, toys, ml_1m).show()


def exp_dropout() -> None:
    param_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    beauty = [
        0.0468,
        0.05156,
        0.0544,
        0.05543,
        0.05535,
        0.05311,
        0.05048,
        0.04618,
        0.04211,
    ]
    toys = [
        0.04274,
        0.04759,
        0.05363,
        0.0518,
        0.0511,
        0.04849,
        0.04425,
        0.04036,
        0.03742,
    ]
    ml_1m = [
        0.1625,
        0.1702,
        0.1781,
        0.1723,
        0.1732,
        0.1661,
        0.1550,
        0.1487,
        0.1342,
    ]
    plot_scores(
        "(b)Dropout", "NDCG@10", param_values, beauty, toys, ml_1m, ax1_ylim_min=0.13
    ).show()


def exp_mask() -> None:
    param_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beauty = [
        0.0510,
        0.0538,
        0.0545,
        0.0554,
        0.0548,
        0.0530,
        0.0514,
        0.0494,
        0.0441,
    ]
    toys = [
        0.0515,
        0.0539,
        0.0527,
        0.0511,
        0.0469,
        0.0451,
        0.0406,
        0.0376,
        0.0335,
    ]
    ml_1m = [0.1721, 0.1759, 0.1755, 0.1791, 0.1761, 0.1723, 0.1649, 0.1550, 0.1357]
    plot_scores(
        "(c)Mask Proportion",
        "NDCG@10",
        param_values,
        beauty,
        toys,
        ml_1m,
        ax1_ylim_min=0.13,
        ax2_ylim_min=0.03,
    ).show()


def exp_layers() -> None:
    param_values = [1, 2, 3, 4]
    beauty = [
        0.0510,
        0.0553,
        0.0558,
        0.0537,
    ]
    toys = [0.0510, 0.0547, 0.0509, 0.0485]
    ml_1m = [0.1734, 0.1735, 0.1706, 0.1651]
    plt = plot_scores(
        "(d)Layers", "NDCG@10", param_values, beauty, toys, ml_1m, ax1_ylim_min=0.13
    )

    plt.xticks(range(1, 5))
    plt.show()


def exp_heads() -> None:
    param_values = [1, 2, 4, 8]
    beauty = [0.0518, 0.0533, 0.0553, 0.0549]
    toys = [0.0480, 0.0501, 0.0547, 0.0505]
    ml_1m = [0.1622, 0.1675, 0.1735, 0.1781]
    plt = plot_scores(
        "(e)Heads",
        "NDCG@10",
        range(len(param_values)),
        beauty,
        toys,
        ml_1m,
        ax1_ylim_min=0.13,
    )
    param_values_str = [str(x) for x in param_values]

    plt.xticks(range(len(param_values)), param_values_str)
    plt.show()


def exp_vd() -> None:
    param_values = [64, 128, 256, 512]
    beauty = [0.05453, 0.05584, 0.05374, 0.0554]
    toys = [0.05235, 0.0540, 0.05485, 0.0502]
    ml_1m = [0.1728, 0.1752, 0.1791, 0.1693]
    plt = plot_scores(
        "(f)Hidden Dimensions of VAE",
        "NDCG@10",
        range(len(param_values)),
        beauty,
        toys,
        ml_1m,
        ax1_ylim_min=0.13,
    )
    param_values_str = [str(x) for x in param_values]

    plt.xticks(range(len(param_values)), param_values_str)
    plt.show()


# exp_length()
# exp_dropout()
# exp_layers()
# exp_heads()
# exp_mask()
# exp_vd()

# %%


# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 16
# dataset = "Toys"


# # Create DataFrame
# if dataset == "Beauty":
#     data = {
#         "TVA": [0.048877, 0.078693, 0.095381, 0.145780, 0.150063],
#         "ContrastVAE": [0.0371, 0.0526, 0.0816, 0.1025, 0.1437],
#         "CBiT": [0.0464, 0.0626, 0.0905, 0.1255, 0.1533],
#     }

# if dataset == "Toys":
#     data = {
#         "TVA": [0.05219, 0.06015, 0.07550, 0.08367, 0.09926],
#         "ContrastVAE": [0.04650, 0.04320, 0.03703, 0.04669, 0.05921],
#         "CBiT": [0.04842, 0.04096, 0.04288, 0.05942, 0.08643],
#     }

# df = pd.DataFrame(data)

# # Bar width
# barWidth = 0.175

# # Set position of bar on X axis
# r1 = np.arange(len(df["TVA"]))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# # Make the plot
# plt.figure(figsize=(5, 5))
# plt.bar(r1, df["TVA"], color="#528ef3", width=barWidth, label="TVA")
# plt.bar(r2, df["ContrastVAE"], color="#ea4335", width=barWidth, label="ContrastVAE")
# plt.bar(r3, df["CBiT"], color="#fab70e", width=barWidth, label="CBiT")


# # Adding xticks
# plt.xlabel(r"\textbf{" + dataset + r"}", fontweight="bold")
# plt.xticks(
#     [r + barWidth for r in range(len(df["TVA"]))],
#     ["$\leq$10", "$[$10, 20$]$", "$[$20, 30$]$", "$[$30, 40$]$", "${>}$ 40"],
# )


# plt.ylabel("NDCG@10")
# # plt.legend()

# plt.show()

# %%


# %%
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


plot_weight()
plot_weight2()

# %%
