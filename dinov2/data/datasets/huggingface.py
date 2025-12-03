# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Tuple
import logging

from datasets import load_dataset
from PIL import Image
import torch
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov2")

class HuggingFaceDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.dataset_name = root # Reusing root as dataset name for consistency with other datasets
        
        logger.info(f"Loading HuggingFace dataset: {self.dataset_name}, split: {self.split}")
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Loaded {len(self.dataset)} samples from {self.dataset_name}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = self.dataset[index]
        
        # Assumes 'image' column. Modify if needed.
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        target = 0 # Dummy target for SSL

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.dataset)
