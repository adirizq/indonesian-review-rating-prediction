import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def save_graph(log_path, model_name, save_path):
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

    fig.suptitle(f'{model_name}\nTest Accuracy: {test_acc}%, Test Loss: {test_loss}\n\n', fontsize='16')

    plt.tight_layout()
    plt.savefig(f'{save_path}/{directory_list[0]["ver"]}.png')

    