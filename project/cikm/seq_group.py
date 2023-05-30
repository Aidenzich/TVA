# %%
import matplotlib.pyplot as plt

dataset = "Beauty"
dataset = "Toys"

# Create DataFrame
if dataset == "Beauty":
    data = {
        "TVaR": [0.048877, 0.078693, 0.095381, 0.145780, 0.150063],
        "ContrastVAE": [0.0371, 0.0526, 0.0816, 0.1025, 0.1437],
        "CBiT": [0.0464, 0.0626, 0.0905, 0.1255, 0.1533],
    }

if dataset == "Toys":
    data = {
        "TVaR": [0.05219, 0.06015, 0.07550, 0.08367, 0.09926],
        "ContrastVAE": [0.04650, 0.04320, 0.03703, 0.04669, 0.05921],
        "CBiT": [0.04842, 0.04096, 0.04288, 0.05942, 0.08643],
    }


def draw_seqlen_group_hist(data):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(6, 6))

    # settings
    colors = [
        "#4285f4",
        "#ea4335",
        "#fbbc05",
        "#34a853",
    ]

    barWidth = 0.175

    # Create Bars
    r_data = {}
    datalen = len(data[list(data.keys())[0]])
    for idx, d in enumerate(data):
        r_data[d] = [x + idx * barWidth for x in range(datalen)]
        plt.bar(
            r_data[d], data[d], width=barWidth, label=d, color=colors[idx % len(colors)]
        )

    plt.xlabel(r"\textbf{" + dataset + r"}", fontweight="bold")
    plt.xticks(
        [r + barWidth for r in range(datalen)],
        ["$\leq$10", "$[$10, 20$]$", "$[$20, 30$]$", "$[$30, 40$]$", "${>}$ 40"],
    )

    plt.ylabel("NDCG@10")
    plt.legend()
    plt.show()


draw_seqlen_group_hist(data)
