import torch
import numpy as np
from torch.utils.data import DataLoader
from data.process_data import make_loaders
from models.majority_classifier import MajorityClassModel
# more models to be imported here
from utils.evaluate import evaluate_model


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model() -> dict:
    """Initialize the models for training and evaluation. Hyperparameters also set here."""
    return {
        'majority': MajorityClassModel(),
        'logistic_regression': # to be implemented
        'cnn': # to be implemented
    }


def run_experiment(setting_name: str, train_loader: DataLoader, val_loader: DataLoader,
                   test_loaders: DataLoader, test_names: str) -> dict:
    """Run the full pipeline for training and evaluating the models on the specified setting.
    
    :param setting_name: name of the experimental setting (cross or intra).
    :param train_loader: training set.
    :param val_loader: validation set.
    :param test_loaders: test sets.
    :param test_names: names for the specific test sets.
    """
    print(f"\n{'=' * 60}\n{setting_name}\n{'=' * 60}")
    results = {}
    for model_name, model in build_model().items():
        print(f"\n>>> Training {model_name} model")
        history = model.fit(train_loader, val_loader)

        results[model_name] = {'history': history, 'test_results': {}}
        for tname, tloader in zip(test_names, test_loaders):
            results[model_name]['test_results'][tname] = evaluate_model(
                model, tloader, name=f"{model_name} on {tname}"
            )
    return results


if __name__ == "__main__":
    set_seed(42)
    data_path = "data/Final_Project_data/"

    # intra
    intra_train, intra_val, intra_tests = make_loaders(
        train_folder=f"{data_path}Intra/train",
        test_folders=[f"{data_path}Intra/test"],
        norm_method='zscore'
    )
    intra_results = run_experiment(
        setting_name="Intra-subject",
        train_loader=intra_train,
        val_loader=intra_val,
        test_loaders=intra_tests,
        test_names=["intra-test"]
    )

    # cross
    cross_train, cross_val, cross_tests = make_loaders(
        train_folder=f"{data_path}Cross/train",
        test_folders=[f"{data_path}Cross/test"],
        norm_method='zscore'
    )
    cross_results = run_experiment(
        setting_name="Cross-subject",
        train_loader=cross_train,
        val_loader=cross_val,
        test_loaders=cross_tests,
        test_names=["cross-test1", "cross-test2", "cross-test3"]
    )
