import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("hyperparameter_results.csv")

plt.figure(figsize=(8,5))

lr_acc = df.groupby("learning_rate")["accuracy"].mean()

plt.plot(
    lr_acc.index.astype(str),
    lr_acc.values,
    marker="o"
)

plt.title("Learning Rate vs Accuracy")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("lr_vs_accuracy.png")
plt.show()

plt.figure(figsize=(8,5))

filter_acc = df.groupby("filters")["accuracy"].mean()

plt.plot(
    filter_acc.index,
    filter_acc.values,
    marker="o"
)

plt.title("Filters vs Accuracy")
plt.xlabel("Number of Filters")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("filters_vs_accuracy.png")
plt.show()

pivot = df.pivot_table(
    values="accuracy",
    index="learning_rate",
    columns="filters"
)

plt.figure(figsize=(8,6))

sns.heatmap(
    pivot,
    annot=True,
    cmap="viridis",
    fmt=".3f"
)

plt.title("Accuracy Heatmap")
plt.savefig("accuracy_heatmap.png")
plt.show()

best = df.sort_values(
    by="accuracy",
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    x=best["filters"].astype(str)
      + "_LR_" +
      best["learning_rate"].astype(str),
    y=best["accuracy"]
)

plt.xticks(rotation=45)
plt.title("Hyperparameter Ranking")
plt.xlabel("Configuration")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("hyperparameter_ranking.png")
plt.show()

print("\nBest Configuration:")
print(best.iloc[0])