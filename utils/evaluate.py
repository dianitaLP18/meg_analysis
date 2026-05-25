from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from models.base_model import AbstractModel


CLASS_NAMES = ['rest', 'task_motor', 'task_story_math', 'task_working_memory']


def evaluate_model(model: AbstractModel, loader: DataLoader, name: str = "test") -> dict:
    """Evaluate the model on the test set.

    :param model: the model to evaluate.
    :param loader: the test data loader.
    :param name: printing purposes only.
    :return: a dictionary containing the evaluation metrics.
    """
    preds, labels = model.predict(loader)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, zero_division=0, digits=4
    )
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(report)
    return {'accuracy': acc, 'confusion_matrix': cm, 'report': report, 'y_true': labels,
        'y_pred': preds}
