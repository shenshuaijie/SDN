import os
from functools import partial

import tonic
import torch
from torch import nn
import torchvision
from tonic import DiskCachedDataset, SlicedDataset
from tonic.slicers import SliceByTime
from torchvision.transforms import RandomPerspective, RandomResizedCrop, RandomRotation
from src.dataloaders.base import SequenceDataset


class MergeEvents:
    def __init__(self, method: str = "mean", flatten: bool = True):
        assert method in ["mean", "diff", "bool", "none"], "Unknown Method"
        self.method = method
        self.flatten = flatten

    def __call__(self, data):
        if self.method == "mean":
            data = torch.mean(data.type(torch.float), dim=1)
        elif self.method == "diff":
            data = data[:, 0, ...] - data[:, 1, ...]
        elif self.method == "bool":
            data = torch.where(data > 1, 1, 0)
        else:
            pass

        if self.flatten:
            return data.reshape((data.size(0), -1))
        else:
            return data


class Gesture(SequenceDataset):
    _name_ = "gesture"
    d_input = 64
    d_output = 11
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,
            "frame_time": 25,
            "min_time_window": 1.7 * 1e6,  # 1.7 s
            "overlap": 0,
            "tr_str": "toframe",
            "event_agg_method": "mean",
            "train_data_max": 19.0,
            "test_data_max": 18.5,
            "flatten": False,
        }

    def setup(self):
        frame_transform_time = tonic.transforms.ToFrame(
            sensor_size=tonic.datasets.DVSGesture.sensor_size,
            time_window=self.frame_time * 1000,
            include_incomplete=False,
        )

        train_dataset = tonic.datasets.DVSGesture(
            save_to=os.path.join(self.data_dir, "train"),
            train=True,
            transform=None,
            target_transform=None,
        )
        test_dataset = tonic.datasets.DVSGesture(
            save_to=os.path.join(self.data_dir, "test"),
            train=False,
            transform=None,
            target_transform=None,
        )

        metadata_path = (
            f"_{self.min_time_window}_{self.overlap}_{self.frame_time}_" + self.tr_str
        )
        slicer_by_time = SliceByTime(
            time_window=self.min_time_window,
            overlap=self.overlap,
            include_incomplete=False,
        )
        train_dataset_timesliced = SlicedDataset(
            train_dataset,
            slicer=slicer_by_time,
            transform=frame_transform_time,
            metadata_path=None,
        )
        test_dataset_timesliced = SlicedDataset(
            test_dataset,
            slicer=slicer_by_time,
            transform=frame_transform_time,
            metadata_path=None,
        )

        if self.event_agg_method == "none" or self.event_agg_method == "mean":
            print(f"Max train value: {self.train_data_max}")
            print(f"Max test value: {self.test_data_max}")
            trian_norm_transform = torchvision.transforms.Lambda(
                lambda x: x / self.train_data_max
            )
            test_norm_transform = torchvision.transforms.Lambda(
                lambda x: x / self.test_data_max
            )
        else:
            trian_norm_transform = None
            test_norm_transform = None

        reduce_transform = torchvision.transforms.Compose(
            [
                partial(nn.functional.max_pool2d, kernel_size=2),
                MergeEvents(method=self.event_agg_method, flatten=self.flatten),
                lambda x: x.view(-1, x.size(-1)),
            ]
        )

        train_post_cache_transform = tonic.transforms.Compose(
            [
                trian_norm_transform,
                torch.tensor,
                RandomResizedCrop(
                    tonic.datasets.DVSGesture.sensor_size[:-1],
                    scale=(0.6, 1.0),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                RandomPerspective(),
                RandomRotation(25),
                reduce_transform,
            ]
        )

        test_post_cache_transform = tonic.transforms.Compose(
            [torch.tensor, test_norm_transform, reduce_transform]
        )

        train_cached_dataset = DiskCachedDataset(
            train_dataset_timesliced,
            transform=train_post_cache_transform,
            cache_path=os.path.join(self.data_dir, "diskcache_train" + metadata_path),
        )

        cached_test_dataset_time = DiskCachedDataset(
            test_dataset_timesliced,
            transform=test_post_cache_transform,
            cache_path=os.path.join(self.data_dir, "diskcache_test" + metadata_path),
        )

        self.collate_fn = tonic.collation.PadTensors(batch_first=True)

        self.dataset_train = train_cached_dataset
        self.dataset_test = cached_test_dataset_time
        self.split_train_val(self.val_split)


if __name__ == "__main__":
    gesture = Gesture("gesture", data_dir="../data/dvs_gesture")
    gesture.setup()
    for data in gesture.dataset_train:
        print(data[0].shape)
        break
    for data in gesture.dataset_test:
        print(data[0].shape)
        break
    for data in gesture.dataset_val:
        print(data[0].shape)
        break
