import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import yaml
from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasouls_antispoof.class_mapping import class_mapping
from datasouls_antispoof.dataloaders import ClassificationDataset

image_path = Path(os.environ["IMAGE_PATH"])


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class AntiSpoof(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = object_from_dict(self.config.model)
        self.loss = object_from_dict(self.config.loss)
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, batch):
        return self.model(batch)

    def setup(self, stage=0):  # pylint: disable=W0613
        self.train_samples = []
        self.val_samples = []

        for class_path in image_path.glob("*"):
            class_name = class_path.name
            class_id = class_mapping[class_name]

            person_paths = sorted(class_path.glob(f"{class_name}_*"))

            num_train = int(len(person_paths) * (1 - self.config.val_split))

            for person_path in person_paths[:num_train]:
                self.train_samples += [(x, class_id) for x in person_path.glob("*.png")]

            for person_path in person_paths[num_train:]:
                self.val_samples += [(x, class_id) for x in person_path.glob("*.png")]

    def train_dataloader(self):
        train_aug = from_dict(self.config.train_aug)

        if "epoch_length" not in self.config.train_parameters:
            epoch_length = None
        else:
            epoch_length = self.config.train_parameters.epoch_length

        result = DataLoader(
            ClassificationDataset(self.train_samples, train_aug, epoch_length),
            batch_size=self.config.train_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.config.val_aug)

        result = DataLoader(
            ClassificationDataset(self.val_samples, val_aug, length=None),
            batch_size=self.config.val_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))
        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.config["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.config.scheduler, optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        features = batch["features"]
        targets = batch["targets"]

        logits = self.forward(features)

        loss = self.loss(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log(
            "train_acc", self.train_accuracy(logits, targets), on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_id):  # pylint: disable=W0613
        features = batch["features"]
        targets = batch["targets"].long()

        logits = self.forward(features)
        loss = self.loss(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "val_acc", self.val_accuracy(logits, targets), on_step=False, on_epoch=True, prog_bar=False, logger=True
        )


def main():
    args = get_args()

    with open(args.config_path) as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    pl.trainer.seed_everything(config.seed)

    pipeline = AntiSpoof(config)

    Path(config.checkpoint_callback.filepath).mkdir(exist_ok=True, parents=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = object_from_dict(
        config.trainer,
        logger=WandbLogger(config.experiment_name),
        checkpoint_callback=object_from_dict(config.checkpoint_callback),
        callbacks=[lr_monitor],
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
