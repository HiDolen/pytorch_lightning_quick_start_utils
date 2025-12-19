from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split


def get_train_val_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
    train_shuffle: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = None,
):
    """
    从 dataset 中划分出训练集和验证集，返回 DataLoader。

    Args:
        dataset: 指定 Dataset 对象。若传入不止一个 Dataset 对象，则会取前两个 分别作为 train 和 val
        batch_size: batch 大小
        num_workers: 指定 num_workers 参数
        pin_memory: 指定 pin_memory 参数
        drop_last: 指定 drop_last 参数
        collate_fn: 指定 collate_fn 参数
        train_shuffle: 对于 train_loader 是否 shuffle 打乱顺序
        persistent_workers: 指定 persistent_workers 参数
        prefetch_factor: 指定 prefetch_factor 参数。num_workers=0 时强制自动为 None
    """

    def create_dataloader(subset, shuffle):
        return DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=shuffle,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    if isinstance(dataset, (list, tuple)):
        train_dataset, val_dataset = dataset[:2]
        train_loader = create_dataloader(train_dataset, shuffle=train_shuffle)
        val_loader = create_dataloader(val_dataset, shuffle=False)
        return train_loader, val_loader

    train_loader = create_dataloader(dataset, shuffle=train_shuffle)
    val_loader = None

    return train_loader, val_loader
