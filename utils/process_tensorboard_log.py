import os
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def save_graph(log_path, model_name, save_path, batch, lr):
    data = pd.DataFrame({"metric": [], "value": [], "step": []})

    directory_list = list()
    for root, dirs, files in os.walk(log_path, topdown=False):
        for name in dirs:
            directory_list.append({'dir':os.path.join(root, name), 'ver':name})

    directory_list = sorted(directory_list, key=lambda x: x['ver'], reverse=True)

    try:
        event_acc = EventAccumulator(directory_list[0]['dir'])
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            data = pd.concat([data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(directory_list[0]['dir']))
        traceback.print_exc()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_loss = data.loc[data['metric'] == 'train_loss_epoch']['value']
    train_acc = data.loc[data['metric'] == 'train_acc_epoch']['value']
    val_loss = data.loc[data['metric'] == 'val_loss']['value']
    val_acc = data.loc[data['metric'] == 'val_acc']['value']
    test_loss = data.loc[data['metric'] == 'test_loss']['value'][0]
    test_acc = data.loc[data['metric'] == 'test_acc']['value'][0]

    max_train_acc = round(train_acc.max() * 100, 2)
    min_train_loss = round(train_loss.min(), 2)
    max_val_acc = round(val_acc.max() * 100, 2)
    min_val_loss = round(val_loss.min(), 2)
    test_loss = round(test_loss, 2)
    test_acc = round(test_acc * 100, 2)

    fig, (acc, loss) = plt.subplots(1,2, figsize=(12,6))
    acc.plot(range(1, len(train_acc)+1), train_acc, label='Train Accuracy')
    acc.plot(range(1, len(val_acc)+1), val_acc, label='Validation Accuracy')
    acc.set_title(f'Model Accuracy\nMax Train Accuracy: {max_train_acc}%, Max Validation Accuracy: {max_val_acc}%', fontsize='large', ha='center')
    acc.set_xlabel('Epoch')
    acc.set_ylabel('Accuracy')
    acc.legend()

    loss.plot(range(1, len(train_loss)+1), train_loss, label='Train Loss')
    loss.plot(range(1, len(val_loss)+1), val_loss, label='Validation Loss')
    loss.set_title(f'Model Loss\nMin Train Loss: {min_train_loss}, Min Validation Loss: {min_val_loss}', fontsize='large', ha='center')
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.legend()

    fig.suptitle(f'{model_name}\nbatch = {batch}, lr = {lr}\nTest Accuracy: {test_acc}%, Test Loss: {test_loss}\n\n', fontsize='16')

    plt.tight_layout()
    plt.savefig(f'{save_path}/{directory_list[0]["ver"]}.png')


def extract_data(log_path):
    data = pd.DataFrame({"metric": [], "value": [], "step": []})

    directory_list = list()
    for root, dirs, files in os.walk(log_path, topdown=False):
        for name in dirs:
            directory_list.append({'dir': os.path.join(root, name), 'ver': name})

    directory_list = sorted(directory_list, key=lambda x: x['ver'], reverse=True)

    try:
        event_acc = EventAccumulator(directory_list[0]['dir'])
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            data = pd.concat([data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(directory_list[0]['dir']))
        traceback.print_exc()

    train_loss = data.loc[data['metric'] == 'train_loss_epoch']['value']
    train_acc = data.loc[data['metric'] == 'train_acc_epoch']['value']
    train_f1_macro = data.loc[data['metric'] == 'train_f1_macro_epoch']['value']
    train_f1_weighted = data.loc[data['metric'] == 'train_f1_weighted_epoch']['value']
    train_f1_rating_1 = data.loc[data['metric'] == 'train_f1_rating_1_epoch']['value']
    train_f1_rating_2 = data.loc[data['metric'] == 'train_f1_rating_2_epoch']['value']
    train_f1_rating_3 = data.loc[data['metric'] == 'train_f1_rating_3_epoch']['value']
    train_f1_rating_4 = data.loc[data['metric'] == 'train_f1_rating_4_epoch']['value']
    train_f1_rating_5 = data.loc[data['metric'] == 'train_f1_rating_5_epoch']['value']

    val_loss = data.loc[data['metric'] == 'val_loss']['value']
    val_acc = data.loc[data['metric'] == 'val_acc']['value']
    val_f1_macro = data.loc[data['metric'] == 'val_f1_macro']['value']
    val_f1_weighted = data.loc[data['metric'] == 'val_f1_weighted']['value']
    val_f1_rating_1 = data.loc[data['metric'] == 'val_f1_rating_1']['value']
    val_f1_rating_2 = data.loc[data['metric'] == 'val_f1_rating_2']['value']
    val_f1_rating_3 = data.loc[data['metric'] == 'val_f1_rating_3']['value']
    val_f1_rating_4 = data.loc[data['metric'] == 'val_f1_rating_4']['value']
    val_f1_rating_5 = data.loc[data['metric'] == 'val_f1_rating_5']['value']

    test_loss = data.loc[data['metric'] == 'test_loss']['value'][0]
    test_acc = data.loc[data['metric'] == 'test_acc']['value'][0]
    test_f1_macro = data.loc[data['metric'] == 'test_f1_macro']['value'][0]
    test_f1_weighted = data.loc[data['metric'] == 'test_f1_weighted']['value'][0]
    test_f1_rating_1 = data.loc[data['metric'] == 'test_f1_rating_1']['value'][0]
    test_f1_rating_2 = data.loc[data['metric'] == 'test_f1_rating_2']['value'][0]
    test_f1_rating_3 = data.loc[data['metric'] == 'test_f1_rating_3']['value'][0]
    test_f1_rating_4 = data.loc[data['metric'] == 'test_f1_rating_4']['value'][0]
    test_f1_rating_5 = data.loc[data['metric'] == 'test_f1_rating_5']['value'][0]

    train_data = pd.DataFrame({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'train_f1_rating_1': train_f1_rating_1,
        'train_f1_rating_2': train_f1_rating_2,
        'train_f1_rating_3': train_f1_rating_3,
        'train_f1_rating_4': train_f1_rating_4,
        'train_f1_rating_5': train_f1_rating_5,
    })

    val_data = pd.DataFrame({
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'val_f1_rating_1': val_f1_rating_1,
        'val_f1_rating_2': val_f1_rating_2,
        'val_f1_rating_3': val_f1_rating_3,
        'val_f1_rating_4': val_f1_rating_4,
        'val_f1_rating_5': val_f1_rating_5,
    })

    test_data = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_rating_1': test_f1_rating_1,
        'test_f1_rating_2': test_f1_rating_2,
        'test_f1_rating_3': test_f1_rating_3,
        'test_f1_rating_4': test_f1_rating_4,
        'test_f1_rating_5': test_f1_rating_5,
    }

    return train_data, val_data, test_data

def save_comparison_graph(models, graph_name):
    for model in models:
        train_data, val_data, test_data = extract_data(model['dir'])
        model.update(train_loss=train_data['train_loss'], train_acc=train_data['train_acc'], val_loss=val_data['val_loss'], val_acc=val_data['val_acc'], test_loss=test_data['test_loss'], test_acc=test_data['test_acc'])

    fig, (loss, acc) = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    for model in models:
        loss[0].plot(range(1, len(model['train_loss'])+1), model['train_loss'], label=model['name'])
    loss[0].set_title('Model Train Loss', fontsize='large', ha='center')
    loss[0].set_xlabel('Epoch')
    loss[0].set_ylabel('Train Loss')
    loss[0].legend()

    for model in models:
        loss[1].plot(range(1, len(model['val_loss'])+1), model['val_loss'], label=model['name'])
    loss[1].set_title('Model Validation Loss', fontsize='large', ha='center')
    loss[1].set_xlabel('Epoch')
    loss[1].set_ylabel('Validation Loss')
    loss[1].legend()

    for model in models:
        acc[0].plot(range(1, len(model['train_acc'])+1), model['train_acc'], label=model['name'])
    acc[0].set_title('Model Train Accuracy', fontsize='large', ha='center')
    acc[0].set_xlabel('Epoch')
    acc[0].set_ylabel('Train Accuracy')
    acc[0].legend()

    for model in models:
        acc[1].plot(range(1, len(model['val_acc'])+1), model['val_acc'], label=model['name'])
    acc[1].set_title('Model Validation Accuracy', fontsize='large', ha='center')
    acc[1].set_xlabel('Epoch')
    acc[1].set_ylabel('Validation Accuracy')
    acc[1].legend()

    fig.suptitle(graph_name, fontsize='16')
    fig.subplots_adjust(hspace=0.3)

    plt.savefig(f'results/comparison/{graph_name}.png')
    