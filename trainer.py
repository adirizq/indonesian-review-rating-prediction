import pytorch_lightning as pl
import importlib
import argparse

from utils.process_tensorboard_log import save_graph
from utils.preprocess import Word2VecDataModule, BERTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from textwrap import dedent


if __name__ == '__main__':
    # Seed for reproducible results
    print("")
    pl.seed_everything(42, workers=True)

    # Arguments for training
    parser = argparse.ArgumentParser(description='Test parser')
    parser.add_argument('-m', '--model', choices=['LSTM', 'CNN 1D', 'CNN 2D', 'BERT', 'BERT CNN 1D', 'BERT CNN 2D'], required=True, help='Model choices to train')
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, help='Learning rate, default [2e-5] for bert-based models and [1e-3] for non-bert-based models')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size, default [32] for bert-based models and [128] for non-bert-based models')
    parser.add_argument('-l', '--max_length', type=int, default=100, required=False, help='Input max length, default [100]')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, required=False, help='Dropout, default [0.5]')

    args = parser.parse_args()
    config = vars(args)

    # Get arguments values
    model_name = config['model']
    max_length = config['max_length']
    dropout = config['dropout']
    batch_size = config['batch_size'] if config['batch_size'] is not None else 128 if model_name in ['LSTM', 'CNN 1D', 'CNN 2D'] else 32
    learning_rate = config['learning_rate'] if config['learning_rate'] is not None else 1e-3 if model_name in ['LSTM', 'CNN 1D', 'CNN 2D'] else 2e-5

    print(dedent(f'''
    -----------------------------------
     Training Information        
    -----------------------------------
     Name                | Value       
    -----------------------------------
     Model Name          | {model_name}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     Input Max Length    | {max_length}
     Dropout             | {dropout}   
    -----------------------------------
    '''))

    # Create model path and model class name
    model_path = model_name.lower().replace(' ', '_')
    model_class_name = model_name.replace(' ', '')

    # Initialize data module and model
    if model_name in ['LSTM', 'CNN 1D', 'CNN 2D']:
        data_module = Word2VecDataModule(max_len=max_length, batch_size=batch_size)
        weigths, vocab_size, embedding_size = data_module.word_embedding()

        ModelClass = getattr(importlib.import_module(f'models.{model_path}'), model_class_name)
        model = ModelClass(word_embedding_weigth=weigths, embedding_size=embedding_size, learning_rate=learning_rate, dropout=dropout)
    else:
        data_module = BERTDataModule(max_len=max_length, batch_size=batch_size)

        ModelClass = getattr(importlib.import_module(f'models.{model_path}'), model_class_name)
        model = ModelClass(learning_rate=learning_rate, dropout=dropout)

    # Initialize callbacks and progressbar
    tensor_board_logger = TensorBoardLogger('logs', name=f'{model_path}/batch={batch_size}_lr={learning_rate}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{model_path}/batch={batch_size}_lr={learning_rate}', monitor='val_loss')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)
    tqdm_progress_bar = TQDMProgressBar()

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir=f'./checkpoints/{model_path}/batch={batch_size}_lr={learning_rate}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=tensor_board_logger,
        log_every_n_steps=5,
        deterministic=True  # To ensure reproducible results
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

    save_graph(f'logs/{model_path}/batch={batch_size}_lr={learning_rate}', model_name, f'results/{model_path}/batch={batch_size}_lr={learning_rate}', batch_size, learning_rate)
