import torch
from datasets.cityscapes import Cityscapes


class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_length,
                 image_size,
                 label_downsample_rate,
                 image_downsample_rate,
                 num_classes=19):
        super().__init__()
        self.dataset_length = dataset_length
        self.image_size = image_size
        self.label_downsample_rate = 1. / label_downsample_rate
        self.image_downsample_rate = 1. / image_downsample_rate
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        image_size = (3, int(self.image_size[1] * self.image_downsample_rate),
                      int(self.image_size[1] * self.image_downsample_rate))
        label_size = (self.num_classes, int(self.image_size[0] * self.label_downsample_rate),
                      int(self.image_size[1] * self.label_downsample_rate))
        return torch.rand(image_size), torch.rand(label_size)


def load_debug_dataset(train_image_size,
                       test_image_size,
                       train_batch_size,
                       test_batch_size,
                       workers,
                       train_label_downsample_rate,
                       train_image_downsample_rate,
                       test_label_downsample_rate,
                       test_image_downsample_rate,
                       num_classes=19):
    train_dataset = DebugDataset(
        dataset_length=train_batch_size,
        image_size=train_image_size,
        label_downsample_rate=train_label_downsample_rate,
        image_downsample_rate=train_image_downsample_rate,
        num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers)

    test_dataset = DebugDataset(
        dataset_length=test_batch_size,
        image_size=test_image_size,
        label_downsample_rate=test_label_downsample_rate,
        image_downsample_rate=test_image_downsample_rate,
        num_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers)

    return {'train': train_loader, 'test': test_loader}


def load_cityscapes(train_batch_size,
                    test_batch_size,
                    train_crop_size,
                    test_crop_size,
                    dataset_root,
                    workers,
                    train_label_downsample_rate,
                    train_image_downsample_rate,
                    test_label_downsample_rate,
                    test_image_downsample_rate
                    ):
    train_dataset = Cityscapes(
        dataset_root=dataset_root,
        list_path='datasets/list/cityscapes/train.lst',
        num_samples=None,
        num_classes=19,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=2048,
        crop_size=train_crop_size,
        label_downsample_rate=train_label_downsample_rate,
        image_downsample_rate=train_image_downsample_rate,
        scale_factor=16)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True)

    test_dataset = Cityscapes(
        dataset_root=dataset_root,
        list_path='datasets/list/cityscapes/val.lst',
        num_samples=0,
        num_classes=19,
        multi_scale=False,
        flip=False,
        ignore_label=-1,
        base_size=2048,
        crop_size=test_crop_size,
        label_downsample_rate=test_label_downsample_rate,
        image_downsample_rate=test_image_downsample_rate)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    return {'train': train_loader, 'test': test_loader}


def load_dataset(dataset,
                 train_batch_size,
                 test_batch_size,
                 train_crop_size,
                 test_crop_size,
                 dataset_root,
                 workers,
                 train_label_downsample_rate,
                 train_image_downsample_rate,
                 test_label_downsample_rate,
                 test_image_downsample_rate):
    if dataset == 'cityscapes':
        return load_cityscapes(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_crop_size=train_crop_size,
            test_crop_size=test_crop_size,
            dataset_root=dataset_root,
            workers=workers,
            train_label_downsample_rate=train_label_downsample_rate,
            train_image_downsample_rate=train_image_downsample_rate,
            test_label_downsample_rate=test_label_downsample_rate,
            test_image_downsample_rate=test_image_downsample_rate), 19
    elif dataset == 'debug':
        return load_debug_dataset(
            train_image_size=train_crop_size,
            test_image_size=test_crop_size,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            workers=workers,
            train_label_downsample_rate=train_label_downsample_rate,
            train_image_downsample_rate=train_image_downsample_rate,
            test_label_downsample_rate=test_label_downsample_rate,
            test_image_downsample_rate=test_image_downsample_rate,
            num_classes=1), 1
