import pandas as pd
import torch
import numpy as np
from data.process_data import make_loaders
from models.cnn import CNNModel
from utils.evaluate import evaluate_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_single_experiment(lr, dropout, num_filters):

    print("=" * 60)
    print(f"LR={lr} | Dropout={dropout} | Filters={num_filters}")
    print("=" * 60)

    train_loader, val_loader, test_loaders = make_loaders(
        train_folder="data/Final_Project_data/Intra/train",
        test_folders=["data/Final_Project_data/Intra/test"],
        norm_method="zscore"
    )

    model = CNNModel(
        lr=lr,
        base_filters= num_filters
    )

    history = model.fit(train_loader, val_loader)

    result = evaluate_model(
        model,
        test_loaders[0],
        name="CNN"
    )

    return result["accuracy"]

def main():

    set_seed()

    learning_rates = [0.01, 0.001, 0.0001]
    dropouts = [0.5]
    filters = [32, 64, 128]

    results = []

    for lr in learning_rates:
        for dropout in dropouts:
            for f in filters:

                try:
                    acc = run_single_experiment(lr, dropout, f)

                    results.append({
                        "learning_rate": lr,
                        "dropout": dropout,
                        "filters": f,
                        "accuracy": acc
                    })

                except Exception as e:
                    print("Error:", e)

    df = pd.DataFrame(results)
    df.to_csv("hyperparameter_results.csv", index=False)

    if len(df) > 0:
         print(df.sort_values(by="accuracy", ascending=False))
    else:
         print("No successful experiments.")


if __name__ == "__main__":
    main()