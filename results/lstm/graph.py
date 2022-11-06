import pandas as pd
import matplotlib.pyplot as plt

train_acc = pd.read_csv('results/lstm/data/run-lstm_classifier_version_2-tag-train accuracy.csv')
train_loss = pd.read_csv('results/lstm/data/run-lstm_classifier_version_2-tag-train loss.csv')
validation_acc = pd.read_csv('results/lstm/data/run-lstm_classifier_version_2-tag-validation accuracy.csv')
validation_loss = pd.read_csv('results/lstm/data/run-lstm_classifier_version_2-tag-validation loss.csv')

plt.plot(train_acc['Step'], train_acc['Value'], label='Train Accuracy')
plt.plot(validation_acc['Step'], validation_acc['Value'], label='Validation Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

plt.plot(train_loss['Step'], train_loss['Value'], label='Train Loss')
plt.plot(validation_loss['Step'], validation_loss['Value'], label='Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()