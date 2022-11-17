from utils.process_tensorboard_log import save_comparison_graph

if __name__ == '__main__':
    all_models = [
        {'name': 'LSTM', 'dir': 'logs/lstm'},
        {'name': 'CNN 1D', 'dir': 'logs/cnn_1d'},
        {'name': 'CNN 2D', 'dir': 'logs/cnn_2d'},
        {'name': 'BERT', 'dir': 'logs/bert'},
        {'name': 'BERT CNN 1D', 'dir': 'logs/bert_cnn_1d'},
        {'name': 'BERT CNN 2D', 'dir': 'logs/bert_cnn_2d'},
    ]

    word2vec_cnn_models = [
        {'name': 'CNN 1D', 'dir': 'logs/cnn_1d'},
        {'name': 'CNN 2D', 'dir': 'logs/cnn_2d'},
    ]

    bert_and_bert_cnn_models = [
        {'name': 'BERT', 'dir': 'logs/bert'},
        {'name': 'BERT CNN 1D', 'dir': 'logs/bert_cnn_1d'},
        {'name': 'BERT CNN 2D', 'dir': 'logs/bert_cnn_2d'},
    ]

    save_comparison_graph(all_models, 'All Models')
    save_comparison_graph(word2vec_cnn_models, 'Word2Vec + CNN Models')
    save_comparison_graph(bert_and_bert_cnn_models, 'BERT and BERT CNN Models')
