import os
import cv2
import torch
import argparse
import numpy as np
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils import data
from torch.nn.functional import one_hot
from torch.optim import lr_scheduler

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from albumentations.pytorch import ToTensorV2


parser = argparse.ArgumentParser(description="Segmentation model runner")
parser.add_argument("--train", action='store_true', help="Flag for Training model", default=False)
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to load into model", default=None)
parser.add_argument("--predict", action='store_true', help="Flag for just predicting from model (exclusive from train)")
parser.add_argument("--dataPath", type=str, help="Path to find data. If training should have train and val subfolders, if predicting should have subfolder called 'to_predict'")

args = parser.parse_args()

assert args.train == False or args.predict == False, "Cannot both train and predict. Pick one."


class wing_dataset(data.Dataset):
    def __init__(self, dataPath='wings', _set='train', transform = None, augment=None, predict=False):
        self.set = _set
        self.dataPath = dataPath
        self.transform = transform # This is transofrmations like rotation, resizing that should be applied to both the mask and input image
        self.augment = augment # This is tranformations only applied to the input image ie, color jitter, contrast, brightness, etc.
        self.predict = predict
        self.to_tensor = A.Compose([ToTensorV2()])

        self.root = Path(f'{self.dataPath}/{self.set}')
        self.image_path = self.root / 'images'
        if not predict:
            self.mask_path = self.root / 'masks'
        
        self.img_list = self.get_filenames(self.image_path)
        if not predict:
            self.mask_list = self.get_filenames(self.mask_path)
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img =  cv2.imread(self.img_list[idx])
        orig_shape = img.shape
        fp = self.img_list[idx]
        if self.predict:
            mask = None
        else:
            mask =  cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)

        
        if self.transform:
            if self.predict:
                transformed = self.transform(image=img)
            else:
                transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed.get('mask', None)
            if mask is not None:
                mask = self.to_tensor(image=mask)['image']
                

        if self.augment:
            img = self.augment(image = img)['image']
        img = self.to_tensor(image=img)['image']

        if self.predict:
            return {'image': img, 'orig_shape': orig_shape, 'fp':fp}
        else:
            return {'image': img, 'mask': mask, 'orig_shape': orig_shape, 'fp':fp}
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class WingSegmentationExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = smp.create_model(
            "unetplusplus",
            encoder_name="resnext50_32x4d",
            in_channels=3,
            classes=7,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params("resnext50_32x4d")
        self.number_of_classes = 7
        #self.model = models.segmentation.fcn_resnet50(num_classes=7).cuda()
        #self.model = models.segmentation.deeplabv3_resnet50(num_classes=7).cuda()
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    def forward(self, images):
        # Normalize image
        images = (images - self.mean) / self.std
        pred = self.model(images)#['out'] # out for pytorch model
        return pred
    
    def training_step(self, batch, batch_idx):
        images = batch['image'].cuda()
        targets = batch['mask'].cuda().long()
        pred = self.forward(images)
        # Ensure the logits mask is contiguous
        pred = pred.contiguous()
        loss = self.loss_fn(pred, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image'].cuda().float()
        targets = batch['mask'].cuda().long()

        pred = self.forward(images)
        # Ensure the logits mask is contiguous for smp model
        pred = pred.contiguous()
        val_loss = self.loss_fn(pred, targets)

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        images = batch['image'].cuda().float()
        orig_shapes = np.array([x.cpu().detach().numpy() for x in batch['orig_shape']]).T
        fps = batch['fp']
        fns = [x.split('/')[-1] for x in fps]



        preds = self.forward(images)
        # Ensure the logits mask is contiguous for smp model
        preds = preds.contiguous()

        for i in range(len(fns)):
            fn = fns[i]
            pred = preds[i]
            orig_shape = orig_shapes[i]
            new_fp = f"{self.trainer.datamodule.dataPath}/predictions/" + fn

            pred = pred.swapaxes(0,2)
            pred = pred.swapaxes(0,1)
            # Apply softmax to get probabilities for multi-class segmentation
            pred = pred.softmax(dim=2)

            # Convert probabilities to predicted class labels
            pred = pred.argmax(dim=2).long()
            pred = pred.cpu().detach().numpy()
            cv2.imwrite(new_fp, pred.astype('uint8'))





    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_validation_epoch_end(self) -> None:
        self.sample_images(_set='train')
        self.sample_images(_set='val')
    
    def sample_images(self, _set='train'):
        # Get sample reconstruction image
        if _set == 'train':
            batch = next(iter(self.trainer.datamodule.train_dataloader()))
        else:
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
        input, label = batch['image'], batch['mask']
        input = input.cuda().float()
        label = label.cuda()


        #         test_input, test_label = batch
        pred = self.forward(input)
        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = pred.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1).long()

        vutils.save_image(
            pred_mask.unsqueeze(1).float().data,
            os.path.join(
                self.logger.log_dir,
                f"{_set}",
                f"Epoch_{self.current_epoch}_pred.png",
            ),
            normalize=True,
            nrow=12,
        )
        vutils.save_image(
            input.data,
            os.path.join(
                self.logger.log_dir,
                f"{_set}",
                f"Epoch_{self.current_epoch}_input.png",
            ),
            normalize=True,
            nrow=12,
        )
        vutils.save_image(
            label.float().data,
            os.path.join(
                self.logger.log_dir,
                f"{_set}",
                f"Epoch_{self.current_epoch}_mask.png",
            ),
            normalize=True,
            nrow=12,
        )


class SegmentationData(pl.LightningDataModule):
    def __init__(self, dataPath: str = "path/to/dir"):
        super().__init__()
        self.dataPath = dataPath
    
    def train_dataloader(self):
        transform = A.Compose([A.Rotate(25, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Affine(degrees=5,translate_percent=(0.05,0.05),scale=(0.85,1.05),shear=1,cval=255,cval_mask=0,mode=cv2.BORDER_CONSTANT),
                                        A.Resize(270, 270),
                                        A.RandomResizedCrop(width=256,height=256,scale=(0.85,1))
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.ColorJitter(brightness = 0.01, contrast=0.05, saturation=0.05, hue=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = wing_dataset(dataPath=self.dataPath, _set='train', transform=transform, augment=augment)
        return torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    
    def val_dataloader(self):
        transform = A.Compose([A.Rotate(0, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Resize(256, 256)
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = wing_dataset(dataPath=self.dataPath, _set='val', transform=transform, augment=augment)
        return torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)

class PredictSegmentationData(pl.LightningDataModule):
    def __init__(self, dataPath: str = "path/to/dir"):
        super().__init__()
        self.dataPath = dataPath

    def predict_dataloader(self):
        transform = A.Compose([A.Rotate(0, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Resize(256, 256)
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        prediction_dataset = wing_dataset(dataPath=self.dataPath, _set='to_predict', transform=transform, augment=augment, predict=True)
        return torch.utils.data.DataLoader(prediction_dataset, batch_size=8, shuffle=False, num_workers=2)

tb_logger = TensorBoardLogger(
    save_dir='logs',
    name='WingingItUNetpp',
)

# For reproducibility
seed_everything(42, True)
Path(f"{tb_logger.log_dir}/train").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/val").mkdir(exist_ok=True, parents=True)
if args.predict:
    Path(f"{args.dataPath}/predictions").mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    if args.checkpoint:
        experiment = WingSegmentationExperiment.load_from_checkpoint(args.checkpoint)
    else:
        experiment = WingSegmentationExperiment()
    trainer = Trainer(logger=tb_logger,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(
                        save_top_k=2,
                        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                    ),
                ],
                max_epochs=100,
                log_every_n_steps=10)
    
    if args.train:
        data = SegmentationData(args.dataPath)
        trainer.fit(experiment, data)

    if args.predict:
        data = PredictSegmentationData(args.dataPath)
        trainer.predict(experiment, dataloaders=data)
