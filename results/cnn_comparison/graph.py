import pandas as pd
import matplotlib.pyplot as plt


cnn_1d_train_acc = pd.read_csv('results/cnn_1d/data/cnn_1d_classifier_version_0_train_acc.csv')
cnn_1d_train_loss = pd.read_csv('results/cnn_1d/data/cnn_1d_classifier_version_0_train_loss.csv')
cnn_1d_val_acc = pd.read_csv('results/cnn_1d/data/cnn_1d_classifier_version_0_val_acc.csv')
cnn_1d_val_loss = pd.read_csv('results/cnn_1d/data/cnn_1d_classifier_version_0_val_loss.csv')

cnn_2d_train_acc = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_train_acc.csv')
cnn_2d_train_loss = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_train_loss.csv')
cnn_2d_val_acc = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_val_acc.csv')
cnn_2d_val_loss = pd.read_csv('results/cnn_2d/data/cnn_2d_classifier_version_0_val_loss.csv')


cnn_1d_max_train_acc = round(cnn_1d_train_acc['Value'].max() * 100, 2)
cnn_1d_min_train_loss = round(cnn_1d_train_loss['Value'].min(), 2)
cnn_1d_max_val_acc = round(cnn_1d_val_acc['Value'].max() * 100, 2)
cnn_1d_min_val_loss = round(cnn_1d_val_loss['Value'].min(), 2)

cnn_2d_max_train_acc = round(cnn_2d_train_acc['Value'].max() * 100, 2)
cnn_2d_min_train_loss = round(cnn_2d_train_loss['Value'].min(), 2)
cnn_2d_max_val_acc = round(cnn_2d_val_acc['Value'].max() * 100, 2)
cnn_2d_min_val_loss = round(cnn_2d_val_loss['Value'].min(), 2)


# Train Accuracy
plt.plot(range(1, len(cnn_1d_train_acc)+1), cnn_1d_train_acc['Value'], label='1D CNN')
plt.plot(range(1, len(cnn_2d_train_acc)+1), cnn_2d_train_acc['Value'], label='2D CNN')
plt.figtext(.5, .95, 'Model Train Accuracy Comparison', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Max 1D Train Accuracy: {cnn_1d_max_train_acc}%, Max 2D Train Accuracy: {cnn_2d_max_train_acc}%', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Validation Accuracy
plt.plot(range(1, len(cnn_1d_val_acc)+1), cnn_1d_val_acc['Value'], label='1D CNN')
plt.plot(range(1, len(cnn_2d_val_acc)+1), cnn_2d_val_acc['Value'], label='2D CNN')
plt.figtext(.5, .95, 'Model Validation Accuracy Comparison', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Max 1D Validation Accuracy: {cnn_1d_max_val_acc}%, Max 2D Validation Accuracy: {cnn_2d_max_val_acc}%', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Train Loss
plt.plot(range(1, len(cnn_1d_train_loss)+1), cnn_1d_train_loss['Value'], label='1D CNN')
plt.plot(range(1, len(cnn_2d_train_loss)+1), cnn_2d_train_loss['Value'], label='2D CNN')
plt.figtext(.5, .95, 'Model Train Loss Comparison', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Min 1D Train Loss: {cnn_1d_min_train_loss}, Min 2D Train Loss: {cnn_2d_min_train_loss}', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Validation Loss
plt.plot(range(1, len(cnn_1d_val_loss)+1), cnn_1d_val_loss['Value'], label='1D CNN')
plt.plot(range(1, len(cnn_2d_val_loss)+1), cnn_2d_val_loss['Value'], label='2D CNN')
plt.figtext(.5, .95, 'Model Validation Loss Comparison', fontsize='large', ha='center')
plt.figtext(.5, .91, f'Min 1D Validation Loss: {cnn_1d_min_val_loss}, Min 2D Validation Loss: {cnn_2d_min_val_loss}', fontsize='medium', ha='center')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
