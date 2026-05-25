import torch
import numpy as np
import os 
from torch.utils.data import DataLoader
from data.process_data import make_loaders
from models.majority_classifier import MajorityClassModel
from models.logistic_regression import LogisticRegressionModel
from models.cnn import CNNModel
from utils.evaluate import evaluate_model
from utils.plotting_helpers import plot_learning_curves, plot_accuracy_curves, plot_confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model() -> dict:
    """Initialize the models for training and evaluation. Hyperparameters also set here."""
    return {
        'majority': MajorityClassModel(),
        'logistic_regression': LogisticRegressionModel(),
        'cnn': CNNModel()
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
    
    class_names = ['rest', 'task_motor', 'task_story_math', 'task_working_memory']

    for model_name, model in build_model().items():
        print(f"\n>>> Training {model_name} model")
        history = model.fit(train_loader, val_loader)

        results[model_name] = {'history': history, 'test_results': {}}
   
        if model_name == 'cnn' and history and isinstance(history, dict):
            print(f"Generating training curves for {model_name}...")
            
            if 'train_loss' in history and 'val_loss' in history:
                plot_learning_curves(
                    train_losses=history['train_loss'],
                    val_losses=history['val_loss'],
                    save_path=f"learning_curve_{setting_name}_{model_name}.png"
                )
            
            if 'train_acc' in history and 'val_acc' in history:
                plot_accuracy_curves(
                    train_accs=history['train_acc'],
                    val_accs=history['val_acc'],
                    save_path=f"accuracy_{setting_name}_{model_name}.png"
                )

        for tname, tloader in zip(test_names, test_loaders):
            test_res = evaluate_model(model, tloader, name=f"{model_name} on {tname}")
            results[model_name]['test_results'][tname] = test_res

            # generate confusion matrices for ALL models to see their performance
            if test_res and 'y_true' in test_res and 'y_pred' in test_res:
                plot_confusion_matrix(
                    y_true=test_res['y_true'],
                    y_pred=test_res['y_pred'],
                    class_names=class_names,
                    save_path=f"confusion_{setting_name}_{model_name}_{tname}.png"
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
        test_folders=[
            f"{data_path}Cross/test1",
            f"{data_path}Cross/test2",
            f"{data_path}Cross/test3",
        ],
        norm_method='zscore'
    )
    cross_results = run_experiment(
        setting_name="Cross-subject",
        train_loader=cross_train,
        val_loader=cross_val,
        test_loaders=cross_tests,
        test_names=["cross-test1", "cross-test2", "cross-test3"]
    )
