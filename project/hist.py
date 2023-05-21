# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.weight"] = "bold"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16

datasets = [r"\textbf{Beauty}", r"\textbf{Toys}", r"\textbf{ML-1m}"]
BERT4Rec = [0.0508, 0.0497, 0.1712]
wo_WarmUp = [0.0493, 0.0486, 0.1681]
wo_Sliding_Window = [0.0439, 0.0407, 0.1563]
wo_Multi_Mask = [0.0427, 0.0428, 0.1579]

bar_width = 0.15
r1 = np.arange(len(datasets))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

plt.figure(figsize=(8, 4))

plt.bar(r1, BERT4Rec, color="#4285F4", width=bar_width, label="BERT4Rec*")
plt.bar(
    r2,
    wo_WarmUp,
    color="#EA4335",
    width=bar_width,
    label="w/o WarmUp",
)
plt.bar(
    r3,
    wo_Sliding_Window,
    color="#FBBC05",
    width=bar_width,
    label="w/o Sliding Window",
)
plt.bar(
    r4,
    wo_Multi_Mask,
    color="#34A853",
    width=bar_width,
    label="w/o Multi-Mask",
)

# plt.xlabel("Datasets", fontweight="bold", fontsize=15)
plt.xticks([r + bar_width for r in range(len(datasets))], datasets)
plt.ylabel("NDCG@10", fontweight="bold", fontsize=15)
# plt.title("Comparison of Different Architectures", fontweight="bold", fontsize=15)

# Adding the scores on top of the bars
for i in range(len(BERT4Rec)):
    plt.text(x=r1[i] - 0.1, y=BERT4Rec[i] + 0.002, s=BERT4Rec[i], size=10)
    plt.text(x=r2[i] - 0.1, y=wo_WarmUp[i] + 0.003, s=wo_WarmUp[i], size=10)
    plt.text(
        x=r3[i] - 0.1, y=wo_Sliding_Window[i] + 0.010, s=wo_Sliding_Window[i], size=10
    )
    plt.text(x=r4[i] - 0.1, y=wo_Multi_Mask[i] + 0.005, s=wo_Multi_Mask[i], size=10)

plt.legend()
plt.show()
