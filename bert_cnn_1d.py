import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from utils.preprocess_bert import ReviewDataModule
from utils.process_tensorboard_log import save_graph
from models.bert_cnn_1d import BertCNN1D

if __name__ == '__main__':
    pl.seed_everything(99, workers=True)

    data_module = ReviewDataModule(max_len=100, batch_size=32)

    model = BertCNN1D(learning_rate=2e-5)

    tensor_board_logger = TensorBoardLogger('logs', name='bert_cnn_1d')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/bert_cnn_1d', monitor='val_loss')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=5)
    tqdm_progress_bar = TQDMProgressBar()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/bert_cnn_1d",
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=tensor_board_logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

    save_graph('logs/bert_cnn_1d', 'BERT CNN 1D', 'results/bert_cnn_1d')
