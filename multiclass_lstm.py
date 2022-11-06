import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from utils.preprocess_lstm import ReviewDataModule
from models.lstm import LSTM

if __name__ == '__main__':
    data_module = ReviewDataModule(max_len=100, batch_size=128)
    weigths, vocab_size, embedding_size = data_module.word_embedding()

    model = LSTM(
        word_embedding_weigth=weigths,
        embedding_size=embedding_size,
        learning_rate=1e-3
    )

    logger = TensorBoardLogger("logs/logs_lstm", name="lstm_classifier")
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/lstm', save_last=True)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/lstm",
        callbacks=[checkpoint_callback, TQDMProgressBar()],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
    # data = trainer.predict(model=model, datamodule=data_module)
