import numpy as np
import os
import matplotlib.pyplot as plt

train_results_path = '/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/4_RLK_Ours/weights_experiments/weight_density_lrschedule/Fold_2'

train_total_loss = np.load(train_results_path + '/' + 'train_loss.npy')
train_dice_loss = np.load(train_results_path + '/' + 'train_dice_loss.npy')
train_density_loss = np.load(train_results_path + '/' + 'train_density_loss.npy')
val_metrics = np.load(train_results_path + '/' + 'val_metrics.npy')

epochs = np.arange(1, len(train_dice_loss) + 1)

# train_density_loss = train_density_loss * 100

plt.figure(figsize=(10, 10))
plt.plot(epochs, train_total_loss, label="Train Loss", color='blue', linewidth=2)
plt.plot(epochs, train_dice_loss, label="Train Dice Loss", color='green', linewidth=2)
plt.plot(epochs, train_density_loss, label="Train Focal Loss", color='magenta', linewidth=2)
plt.plot(epochs, val_metrics[:, 0], label="Valid Target F1", color='red', linewidth=2)
plt.plot(epochs, val_metrics[:, 1], label="Valid Pixel F1", color='cyan', linewidth=2)
plt.plot(epochs, val_metrics[:, 2], label="Valid Average F1", color='darkorange', linewidth=2)
plt.ylim(0, 1.5)

plt.title("Training Losses and Valid Metrics", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Losses", fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)

# plt.savefig(train_results_path + '/' + "loss_plot.png")

plt.show()