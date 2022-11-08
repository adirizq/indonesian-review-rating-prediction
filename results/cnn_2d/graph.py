import pandas as pd
import matplotlib.pyplot as plt

train_acc = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_train_acc.csv')
train_loss = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_train_loss.csv')
val_acc = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_val_acc.csv')
val_loss = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_val_loss.csv')

max_train_acc = round(train_acc['Value'].max() * 100, 2)
min_train_loss = round(train_loss['Value'].min(), 2)
max_val_acc = round(val_acc['Value'].max() * 100, 2)
min_val_loss = round(val_loss['Value'].min(), 2)

plt.plot(range(1, len(train_acc)+1), train_acc['Value'], label='Train Accuracy')
plt.plot(range(1, len(val_acc)+1), val_acc['Value'], label='Validation Accuracy')
plt.figtext(.5, .95, '2D CNN Model Accuracy', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Max Train Accuracy: {max_train_acc}%, Max Validation Accuracy: {max_val_acc}%', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, len(train_loss)+1), train_loss['Value'], label='Train Loss')
plt.plot(range(1, len(val_loss)+1), val_loss['Value'], label='Validation Loss')
plt.figtext(.5, .95, '2D CNN Model Loss', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Min Train Loss: {min_train_loss}, Min Validation Loss: {min_val_loss}', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
