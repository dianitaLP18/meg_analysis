from data.process_data import make_loaders
from models.cnn import CNNModel
from utils.evaluate import evaluate_model

def main():
    train_loader, val_loader, test_loaders = make_loaders(
        train_folder="data/Final_Project_data/Cross/train",
        test_folders=[
            "data/Final_Project_data/Cross/test1",
            "data/Final_Project_data/Cross/test2",
            "data/Final_Project_data/Cross/test3"
        ],
        norm_method="zscore",
        cross_subject=True,
        num_workers=0     
    )

    model = CNNModel(
        lr=0.0001,
        base_filters=64
    )

    history = model.fit(
        train_loader,
        val_loader
    )

    print("\n" + "=" * 60)
    print("CROSS SUBJECT RESULTS")
    print("=" * 60)

    for i, test_loader in enumerate(test_loaders, start=1):

        result = evaluate_model(
            model,
            test_loader,
            name=f"Cross-Test{i}"
        )

        print(
            f"Cross Test{i} Accuracy: "
            f"{result['accuracy']:.4f}"
        )
        
if __name__ == "__main__":
    main()