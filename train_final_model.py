from data.process_data import make_loaders
from models.cnn import CNNModel
from utils.evaluate import evaluate_model


def main():

    # Load Data
    train_loader, val_loader, test_loaders = make_loaders(
        train_folder="data/Final_Project_data/Intra/train",
        test_folders=["data/Final_Project_data/Intra/test"],
        norm_method="zscore"
    )

    # Best Hyperparameters
    model = CNNModel(
        lr=0.0001,
        base_filters=64
    )

    # Train Model
    history = model.fit(
        train_loader,
        val_loader
    )

    # Evaluate
    result = evaluate_model(
        model,
        test_loaders[0],
        name="Final CNN"
    )

    print("\nFinal Accuracy:", result["accuracy"])


if __name__ == "__main__":
    main()