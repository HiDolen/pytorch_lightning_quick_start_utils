from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def get_train_val_dataloader(
    dataset: Dataset,
    batch_size: int,
    test_size: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
    train_shuffle: bool = True,
):
    """
    从 dataset 中划分出训练集和验证集，返回 DataLoader。

    Args:
        dataset: 指定 Dataset 对象
        batch_size: batch 大小
        test_size: 验证集占比。为 0 时函数只返回 train_loader
        num_workers: 指定 num_workers 参数
        pin_memory: 指定 pin_memory 参数
        drop_last: 指定 drop_last 参数
        collate_fn: 指定 collate_fn 参数
        train_shuffle: 对于 train_loader 是否 shuffle 打乱顺序
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
        )

    train_dataset, val_dataset = train_test_split(dataset, test_size=test_size, random_state=42)

    train_loader = create_dataloader(train_dataset, shuffle=train_shuffle)
    val_loader = create_dataloader(val_dataset, shuffle=False) if test_size != 0 else None

    return train_loader, val_loader
