import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from utils.preprocess_word2vec import ReviewDataModule
from models.cnn_2d import CNN2D

if __name__ == '__main__':
    data_module = ReviewDataModule(max_len=100, batch_size=128)
    weigths, vocab_size, embedding_size = data_module.word_embedding()

    model = CNN2D(
        word_embedding_weigth=weigths,
        embedding_size=embedding_size,
    )

    logger = TensorBoardLogger("logs/logs_cnn_2d", name="cnn_2d_classifier")
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/cnn_2d', save_last=True)
    early_stop_callback = EarlyStopping(monitor='validation loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)
    tqdm_progress_bar = TQDMProgressBar()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/cnn_2d",
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
    # data = trainer.predict(model=model, datamodule=data_module)
