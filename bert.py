import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from utils.preprocess_bert import ReviewDataModule
from models.bert import Bert

if __name__ == '__main__':
    pl.seed_everything(99, workers=True)

    data_module = ReviewDataModule(max_len=100, batch_size=128)

    model = Bert()

    logger = TensorBoardLogger('logs/bert')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/bert', save_last=True)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)
    tqdm_progress_bar = TQDMProgressBar()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/bert",
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
