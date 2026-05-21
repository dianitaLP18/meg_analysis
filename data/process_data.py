import os
import re
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict, defaultdict
from typing import Literal
from sklearn.model_selection import train_test_split


# configuration variables
LABEL_MAP = {
    'rest': 0,
    'task_motor': 1,
    'task_story_math': 2,
    'task_working_memory': 3
}
WINDOW_SIZE = 256
STRIDE = 256
DOWNSAMPLE_FACTOR = 4
NORM_METHOD = Literal['zscore', 'minmax']


# preprocessing functions
def downsample(matrix: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """Downsamples the matrix by selecting every nth column."""
    return matrix[:, ::factor]


def normalise(matrix: np.ndarray, method: NORM_METHOD) -> np.ndarray:
    """Normalises the matrix using the specified method.

    :param matrix: the input matrix to normalise.
    :param method: the normalisation method to use.
    :return: the normalised matrix.
    """
    if method == 'zscore':
        mean = np.mean(matrix, axis=1, keepdims=True)
        std = np.std(matrix, axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (matrix - mean) / std
    elif method == 'minmax':
        min_val = np.min(matrix, axis=1, keepdims=True)
        max_val = np.max(matrix, axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return (matrix - min_val) / range_val

    raise ValueError(f"Unsupported normalisation method: {method}")


def preprocess_matrix(matrix: np.ndarray, method: NORM_METHOD = 'zscore') -> np.ndarray:
    """Applied downsampling and normalisation to the input matrix.

    :param matrix: the raw input matrix to preprocess.
    :param norm_method: the normalisation method to apply after downsampling.
    :return: the preprocessed matrix.
    """
    return normalise(downsample(matrix), method=method).astype(np.float32)


# indexing and dataset
def _get_label_from_filename(filename: str) -> int | None:
    """Extracts the label from the filename based on the LABEL_MAP.

    :param filename: the name of the file to extract the label from.
    :return: the corresponding label integer or None if no match is found.
    """
    return next((v for p, v in LABEL_MAP.items() if filename.startswith(p)), None)


def build_window_index(folder_path: str, filenames: list[str] | None = None) -> list[tuple[str, int, int]]:
    """Scans a folder for .h5 files and builds an
    index of (filename, start_col, end_col) tuples.

    :param folder_path: path to the folder containing .h5 files.
    :param filenames: optional list of filenames to include.
    :return: list of tuples with filename and column indices.
    """
    if filenames is None:
        filenames = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.h5')]

    index = []
    for filename in filenames:
        label = _get_label_from_filename(filename)
        if label is None:
            print(f"[warn] skipping file with unknown label prefix: {filename}")
            continue
        filepath = os.path.join(folder_path, filename)
        # look at the dimension without loading
        with h5py.File(filepath, 'r') as f:
            key = list(f.keys())[0]
            # account for downsampling
            n_timesteps = f[key].shape[1] // DOWNSAMPLE_FACTOR
        for start in range(0, n_timesteps - WINDOW_SIZE + 1, STRIDE):
            index.append((filepath, start, label))
    return index


class MEGWindowDataset(Dataset):
    def __init__(self, index: list[tuple[str, int, int]], norm_method: NORM_METHOD, cache_size: int = 1) -> None:
        """Initialises the dataset with an index of file paths and column ranges.

        :param index: list of tuples (filepath, start_col, label).
        :param cache_files: whether to load all files into memory for faster access.
        """
        self.index = index
        self.norm_method = norm_method
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filepath, start, label = self.index[idx]
        matrix = self._get_matrix(filepath)
        window = matrix[:, start:start + WINDOW_SIZE]
        return torch.from_numpy(window).float(), torch.tensor(label, dtype=torch.long)

    def _get_matrix(self, filepath: str) -> np.ndarray:
        """Loads the matrix from the file, using cache if enabled.

        :param filepath: path to the .h5 file.
        :return: the preprocessed matrix as a NumPy array.
        """
        if filepath in self._cache:
            self._cache.move_to_end(filepath)
            return self._cache[filepath]
        with h5py.File(filepath, 'r') as f:
            key = list(f.keys())[0]
            raw = f[key][()]
        processed = preprocess_matrix(raw, method=self.norm_method)
        self._cache[filepath] = processed
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return processed


# splits
def splitting(folder_path: str, val_fraction: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    """Splits the dataset into training and validation sets based on file paths.

    :param folder_path: path to the folder containing .h5 files.
    :param val_fraction: fraction of data to use for validation.
    :param seed: random seed for reproducibility.
    :return: tuple of (train_files, val_files) lists.
    """
    files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.h5')]
    labels = [_get_label_from_filename(f) for f in files]
    pairs = [(f, l) for f, l in zip(files, labels) if l is not None]
    files, labels = zip(*pairs)
    train_files, val_files = train_test_split(
        list(files), test_size=val_fraction, stratify=labels, random_state=seed
    )
    return train_files, val_files


def subject_aware_split(folder_path: str, val_subject: str | None = None) -> tuple[list[str], list[str]]:
    """Split files so that val_subject's files form the val set.

    :param folder_path: path to the folder containing .h5 files.
    :param val_subject: subject ID to use for validation.
           If None, the subject with the fewest files will be used.
    :return: tuple of (train_files, val_files) lists.
    """
    files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.h5')]
    by_subject: dict[str, list[str]] = defaultdict(list)
    for f in files:
        m = re.search(r'(\d{6})', f)
        if m:
            by_subject[m.group(1)].append(f)

    if val_subject is None:
        val_subject = min(by_subject, key=lambda s: len(by_subject[s]))

    val_files = by_subject[val_subject]
    train_files = [f for sublist_id, sublist in by_subject.items()
                   if sublist_id != val_subject for f in sublist]
    return train_files, val_files


# build loaders
def make_loaders(
        train_folder: str, test_folders: list[str],
        norm_method: NORM_METHOD, batch_size: int = 32,
        val_fraction: float = 0.2, num_workers: int = 2,
        cross_subject: bool = False) -> tuple[DataLoader, DataLoader, list[DataLoader]]:
    """Creates PyTorch DataLoaders for training and validation datasets.

    :param train_folder: path to the training data folder.
    :param test_folders: list of paths to test data folders.
    :param norm_method: normalisation method to apply to the data.
    :param batch_size: number of samples per batch.
    :param val_fraction: fraction of training data to use for validation.
    :param num_workers: number of subprocesses to use for data loading.
    :param cross_subject: whether to use subject-aware splitting for validation.
    :return: tuple of (train_loader, val_loader).
    """
    if cross_subject:
        train_files, val_files = subject_aware_split(train_folder)
    else:
        train_files, val_files = splitting(train_folder, val_fraction)

    train_index = build_window_index(train_folder, filenames=train_files)
    val_index = build_window_index(train_folder, filenames=val_files)

    train_dataset = MEGWindowDataset(train_index, norm_method=norm_method)
    val_dataset = MEGWindowDataset(val_index, norm_method=norm_method)
    test_datasets = [MEGWindowDataset(build_window_index(p), norm_method=norm_method) for p in test_folders]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    for ds in test_datasets]

    assert len(train_index) > 0, f"Empty train index for {train_folder}"
    assert len(val_index) > 0, f"Empty val index for {train_folder}"
    train_labels = {lbl for _, _, lbl in train_index}
    val_labels = {lbl for _, _, lbl in val_index}
    assert train_labels == set(LABEL_MAP.values()), f"Missing classes in train: {set(LABEL_MAP.values()) - train_labels}"
    assert val_labels == set(LABEL_MAP.values()), f"Missing classes in val: {set(LABEL_MAP.values()) - val_labels}"

    return train_loader, val_loader, test_loaders


if __name__ == "__main__":
    base = "data/Final_Project_data/"

    # intra subject split
    intra_train, intra_val, intra_tests = make_loaders(
        train_folder=f"{base}Intra/train",
        test_folders=[f"{base}Intra/test"],
        norm_method='zscore'
    )

    # cross subject split
    cross_train, cross_val, cross_tests = make_loaders(
        train_folder=f"{base}Cross/train",
        test_folders=[f"{base}Cross/test1", f"{base}Cross/test2", f"{base}Cross/test3"],
        norm_method='zscore',
        cross_subject=True
    )
    print(f"Intra train windows: {len(intra_train.dataset)}")
    print(f"Intra val windows: {len(intra_val.dataset)}")
    print(f"Intra test windows: {len(intra_tests[0].dataset)}")
    print("-" * 40)
    print(f"Cross train windows: {len(cross_train.dataset)}")
    print(f"Cross val windows:   {len(cross_val.dataset)}")
    for i, tl in enumerate(cross_tests, 1):
        print(f"Cross test{i} windows: {len(tl.dataset)}")
