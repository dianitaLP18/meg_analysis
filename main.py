import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
from data.process_data import make_loaders
from data.augmentations import AugmentationConfig
from models.majority_classifier import MajorityClassModel
from models.logistic_regression import LogisticRegressionModel
from models.cnn import CNNModel
from models.eegnet import EEGNetModel
from utils.evaluate import evaluate_model
from utils.plotting_helpers import plot_learning_curves, plot_accuracy_curves, plot_confusion_matrix


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SEEDS = [42, 123, 2024]


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_deterministic_models() -> dict:
    """Initialize the deterministic models for training and evaluation."""
    return {
        'majority': MajorityClassModel(),
        'logistic_regression': LogisticRegressionModel()
    }


def build_seeded_models() -> dict:
    """Initialize the seeded models for training and evaluation."""
    return {
        'cnn': CNNModel(),
        'eegnet': EEGNetModel()
    }


def run_one_setting(setting_name: str, train_loader: DataLoader, val_loader: DataLoader,
                    test_loaders: DataLoader, test_names: str, seeds: list[int]) -> dict:
    """Run the full pipeline for training and evaluating the models on the specified setting.

    :param setting_name: name of the experimental setting (cross or intra).
    :param train_loader: training set.
    :param val_loader: validation set.
    :param test_loaders: test sets.
    :param test_names: names for the specific test sets.
    :param seeds: list of random seeds for testing.
    """
    print(f"\n{'=' * 60}\n{setting_name}\n{'=' * 60}")
    accs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    class_names = ['rest', 'task_motor', 'task_story_math', 'task_working_memory']
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # deterministic models
    for name, model in build_deterministic_models().items():
        print(f"\n>>> Training {name} model (single run)")
        model.fit(train_loader, val_loader)
        for tname, tloader in zip(test_names, test_loaders):
            res = evaluate_model(model, tloader, name=f"{name} on {tname}")
            accs[name][tname].append(res['accuracy'])
            if name == 'majority':
                continue
            else:
                plot_confusion_matrix(
                    y_true=res['y_true'], y_pred=res['y_pred'],
                    class_names=class_names,
                    save_path=f"confusion_{setting_name}_{name}_{tname}.png",
                )

    # seeded models
    for seed in seeds:
        print(f"\n>>> Training seeded models with seed {seed}")
        set_seed(seed)
        for name, model in build_seeded_models().items():
            print(f"\n>>> Training {name} model")
            history = model.fit(train_loader, val_loader)
            first_seed = (seed == seeds[0])

            if first_seed and isinstance(history, dict):
                if 'train_loss' in history and 'val_loss' in history:
                    plot_learning_curves(
                        train_losses=history['train_loss'],
                        val_losses=history['val_loss'],
                        save_path=f"learning_curve_{setting_name}_{name}.png",
                    )
                if 'train_acc' in history and 'val_acc' in history:
                    plot_accuracy_curves(
                        train_accs=history['train_acc'],
                        val_accs=history['val_acc'],
                        save_path=f"accuracy_{setting_name}_{name}.png",
                    )

            for tname, tloader in zip(test_names, test_loaders):
                res = evaluate_model(model, tloader, name=f"{name} (s={seed} on {tname}")
                accs[name][tname].append(res['accuracy'])
                if first_seed:
                    plot_confusion_matrix(
                        y_true=res['y_true'], y_pred=res['y_pred'],
                        class_names=class_names,
                        save_path=f"confusion_{setting_name}_{name}_{tname}.png",
                    )

    return accs


def make_table(accs: dict, setting_name: str, test_names: list[str]) -> pd.DataFrame:
    """Create a summary table of the accuracies for each model and test set.

    :param accs: dictionary of accuracies for each model and test set.
    :param setting_name: name of the experimental setting (cross or intra).
    :param test_names: names for the specific test sets.
    :return: a pandas DataFrame summarizing the results.
    """
    rows = []
    for model_name, per_test in accs.items():
        row = {'Setting': setting_name, 'Model': model_name}
        all_means = []
        for tname in test_names:
            arr = np.array(per_test[tname]) * 100
            if len(arr) == 1:
                row[tname] = f"{arr[0]:.2f}"
            else:
                row[tname] = f"{arr.mean():.2f} ± {arr.std(ddof=0):.2f}"
                all_means.append(arr.mean())
        if len(test_names) > 1:
            row['Mean'] = f"{np.mean(all_means):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data_path = "data/Final_Project_data/"

    aug = AugmentationConfig(
        channel_dropout_prob=0.3, channel_dropout_frac=0.05,
        noise_prob=0.3, noise_std=0.03, time_shift_prob=0.5,
        time_shift_max=16, time_mask_prob=0.0, time_mask_max=0,
    )

    # intra
    intra_train, intra_val, intra_tests = make_loaders(
        train_folder=f"{data_path}Intra/train",
        test_folders=[f"{data_path}Intra/test"],
        norm_method='zscore',
        augment_config=aug
    )
    intra_accs = run_one_setting(
        setting_name="Intra-subject",
        train_loader=intra_train,
        val_loader=intra_val,
        test_loaders=intra_tests,
        test_names=["intra-test"],
        seeds=SEEDS
    )

    # cross
    cross_train, cross_val, cross_tests = make_loaders(
        train_folder=f"{data_path}Cross/train",
        test_folders=[
            f"{data_path}Cross/test1",
            f"{data_path}Cross/test2",
            f"{data_path}Cross/test3",
        ],
        norm_method='zscore',
        augment_config=aug,
        cross_subject=True
    )
    cross_accs = run_one_setting(
        setting_name="Cross-subject",
        train_loader=cross_train,
        val_loader=cross_val,
        test_loaders=cross_tests,
        test_names=["cross-test1", "cross-test2", "cross-test3"],
        seeds=SEEDS
    )

    # tables
    intra_table = make_table(intra_accs, setting_name="Intra-subject",
                             test_names=['intra-test'])
    cross_table = make_table(cross_accs, setting_name="Cross-subject",
                             test_names=['cross-test1', 'cross-test2', 'cross-test3'])
    combined = pd.concat([intra_table, cross_table], ignore_index=True)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(combined.to_string(index=False))

    combined.to_csv(os.path.join(ROOT, "results", "results.csv"), index=False)
    with open(os.path.join(ROOT, "results", "results.md"), "w") as f:
        f.write(combined.to_markdown(index=False))
