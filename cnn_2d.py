import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from utils.preprocess_word2vec import ReviewDataModule
from utils.process_tensorboard_log import save_graph
from models.cnn_2d import CNN2D

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    data_module = ReviewDataModule(max_len=100, batch_size=128)
    weigths, vocab_size, embedding_size = data_module.word_embedding()

    model = CNN2D(
        word_embedding_weigth=weigths,
        embedding_size=embedding_size,
    )

    tensor_board_logger = TensorBoardLogger('logs', name='cnn_2d')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/cnn_2d', monitor='val_loss')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)
    tqdm_progress_bar = TQDMProgressBar()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/cnn_2d",
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=tensor_board_logger,
        log_every_n_steps=5,
        deterministic=True
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

    save_graph('logs/cnn_2d', 'Two Dimensional CNN', 'results/cnn_2d')
