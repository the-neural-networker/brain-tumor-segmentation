import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import pandas as pd 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.dataset import BratsDataModule
from src.networks.unet import UNet3d
from src.networks.unetr import UNeTr


def main() -> None: 
    args = get_args() 

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    dm = BratsDataModule(
        train_df=train_df, 
        test_df=test_df,
        is_resize=args.is_resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )

    dm.prepare_data()
    dm.setup()

    if args.model == 'unetr':
        model = UNeTr(
            img_shape=args.img_shape,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )
    elif args.model == 'unet':
        model = UNet3d(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            learning_rate=args.learning_rate
        )

    early_stopping = EarlyStopping('val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        dirpath="../checkpoints/",
        filename=f"{args.model}" + "-{epoch:02d}-{val_loss_epoch:.2f}",
        save_top_k=5,
    )

    checkpoint_callback.FILE_EXTENSION = ".pth.tar"
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(gpus=args.gpus,
                        max_epochs=args.max_epochs, 
                        limit_train_batches=args.limit_train_batches, 
                        limit_val_batches=args.limit_val_batches,
                        limit_test_batches=args.limit_test_batches,
                        callbacks=[early_stopping, checkpoint_callback, lr_monitor]
                    )
                    
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)

def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--train_csv", default="/u/akommala/brain-tumor-segmentation/notebooks/train_data.csv", type=str)
    parser.add_argument("--test_csv", default="/u/akommala/brain-tumor-segmentation/notebooks/test_data.csv", type=str)
    parser.add_argument("--model", default='unetr', type=str)
    parser.add_argument("--is_resize", default=True, type=bool)
    parser.add_argument("--batch_size", default=1, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNet3d.add_model_specific_args(parser)
    remove_argument(parser, '--learning_rate')
    parser = UNeTr.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    main()