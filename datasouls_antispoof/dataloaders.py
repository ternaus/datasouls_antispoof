from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as albu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: albu.Compose,
        length: int = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, class_id = self.samples[idx]

        image = load_rgb(image_path, lib="cv2")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": image_path.stem, "features": tensor_from_rgb_image(image), "targets": class_id}
