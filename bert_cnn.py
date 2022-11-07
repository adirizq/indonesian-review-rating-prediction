import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from utils.preprocess_bert import ReviewDataModule
from models.bert_cnn import BertCNN

if __name__ == '__main__':
    data_module = ReviewDataModule(max_len=100, batch_size=128)

    model = BertCNN()

    logger = TensorBoardLogger("logs/logs_bert_cnn", name="bert_cnn_classifier")
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/bert_cnn', save_last=True)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/bert_cnn",
        callbacks=[checkpoint_callback, TQDMProgressBar()],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
    # data = trainer.predict(model=model, datamodule=data_module)
