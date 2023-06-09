import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

import data
import models

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='')
# parser.add_argument()
parser.add_argument('--path', type=str, help="Path to the saved model.ckpt")
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--num_nodes', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--logdir', type=str, default='./logs')
args = parser.parse_args()

dirname = os.path.dirname(args.path)
descriptor = os.path.basename(dirname)
model_type, train_mode, test_mode, sample = descriptor.split("_")
regular = (sample == "regular")

loader = data.PersonData(args.seq_len, regular)
model = models.build_model(
  model_type=model_type,
  num_nodes=args.num_nodes,
  training_mode=train_mode,
  testing_mode=test_mode,
  batch_size=args.batch_size,
  seq_length=args.seq_len,
  num_x_features=loader.train_x.shape[2],
  regular=regular,
)

# Load the trained weights
model.load_weights(args.path)

all_preds = []
for i in tqdm(range(0, loader.test_x.shape[0], args.batch_size)):
  x, y, t = [tf.convert_to_tensor(arr) for arr in (
    loader.test_x[i:i+args.batch_size],
    loader.test_y[i:i+args.batch_size],
    loader.test_t[i:i+args.batch_size])]

  ### TODO: if test == "adversarial" then we should do an adversarial step here.
  if regular or model_type == "LSTM":
    if test_mode == "adversarial":
      pred = model.adv_call(x, y)
    else:
      pred = model(x)
  else:
    if test_mode == "adversarial":
      pred = model.adv_call((x, t), y)
    else:
      pred = model((x, t))
  all_preds.append(pred)

all_preds = np.concatenate(all_preds, axis=0)
arg_max_preds = np.argmax(all_preds, -1)
# print(loader.test_y.shape)
# print(np.argmax(all_preds, -1).shape)
# print(loader.test_y[3])
# print(np.argmax(all_preds, -1)[3])
# print(loader.test_y[3] == np.argmax(all_preds, -1)[3])
# print("HFIHSOIDHFIHOSHFIHSDOS", loader.test_y[0])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
total_loss = loss_fn(loader.test_y, all_preds)
total_accuracy = np.mean(loader.test_y == arg_max_preds)
accuracies_folder_path = os.path.join(args.logdir, "accuracies")
if not os.path.exists(accuracies_folder_path):
  os.makedirs(accuracies_folder_path)
save_accuracy_path = os.path.join(accuracies_folder_path, descriptor)
with open(save_accuracy_path, "w") as file:
  file.write(str(total_accuracy))
print(f"total_loss: {total_loss} \t total_accuracy: {total_accuracy}")

### Plot any additional plots here -- all variables should be computed and
# ready to visualize: (ground truth = loader.test_y) and predicted (logits) = all_preds
sns.set_style("whitegrid")

def generate_confusion_matrix(preds, labels, num_classes):
  all_preds= []
  all_labels = []
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      all_preds.append(preds[i][j])
      all_labels.append(labels[i][j])
  confusion_matrix = np.array(tf.math.confusion_matrix(all_labels, all_preds, num_classes))
  return confusion_matrix

def plot_confusion_matrix(preds, labels, num_classes):
  confusion_matrix = generate_confusion_matrix(preds, labels, num_classes)
  class_labels = ['State 0', 'State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6']
  model_type, train_mode, test_mode, sample = descriptor.split("_")
  
  # Plot the confusion matrix as a table
  fig, ax = plt.subplots(figsize=(8, 8))
  if train_mode == "nonadversarial":
    train_mode = "standard"
  ax.set_title(f"{train_mode.capitalize()} {model_type} {test_mode.capitalize()} Test On {sample.capitalize()} Dataset", x = 0.48, fontsize = 12, pad=0)
  # ax.set_title(f"{model_type}_{train_mode.capitalize()}_{test_mode.capitalize()}_{sample.capitalize()} Matrix", x = 0.47, pad=0)
  ax.set_xlabel('Predicted Labels', labelpad=0)
  ax.set_ylabel('True Labels', labelpad=0)
  table = ax.table(cellText=confusion_matrix,
                  cellLoc='center',
                  loc='center',
                  colLabels=class_labels,
                  rowLabels=class_labels)

  # Set table properties
  table.scale(3, 4)
  # table.auto_set_font_size(False)
  # table.set_fontsize(14)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.xaxis.set_tick_params(size=0)
  ax.yaxis.set_tick_params(size=0)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Format table cells
  for i in range(len(class_labels)):
    for j in range(len(class_labels)):
      if i == j:
        table[(i+1, j)].set_facecolor('#C1FFC1')  # Set diagonal cells color
      else:
        table[(i+1, j)].set_facecolor('#FFC1C1')  # Set off-diagonal cells color

  for i in range(len(class_labels)):
    table.auto_set_column_width(col=i)

  plt.savefig(os.path.join(args.logdir, descriptor, "confusion_matrix.pdf"))
  plt.savefig(os.path.join(args.logdir, descriptor, "confusion_matrix.png"))
  plt.close()

plot_confusion_matrix(arg_max_preds, loader.test_y, loader.num_classes)

def plot_class_accuracy_bar(preds, labels, num_classes):
  pred_correctness = (loader.test_y == arg_max_preds)
  class_pred_correctness = []
  for c in range(num_classes):
    class_pred_correctness.append([])
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      class_pred_correctness[int(labels[i][j])].append(pred_correctness[i][j])
  class_accuracies = []
  for i in range(num_classes):
    class_accuracies.append(np.mean(class_pred_correctness[i]))
  x = np.arange(len(class_accuracies))
  model_type, train_mode, test_mode, sample = descriptor.split("_")
  plt.bar(x, class_accuracies)
  plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=8)
  plt.xticks(fontsize=8)
  plt.xlabel('Class', fontsize = 10, labelpad = 8)
  plt.ylabel('Accuracy', fontsize = 10, labelpad = 10)
  if train_mode == "nonadversarial":
    train_mode = "standard"
  plt.title(f"{train_mode.capitalize()} {model_type} {test_mode.capitalize()} Test On {sample.capitalize()} Dataset", fontsize = 12, pad = 12)
  plt.savefig(os.path.join(args.logdir, descriptor, "class_accuracies.pdf"))
  plt.savefig(os.path.join(args.logdir, descriptor, "class_accuracies.png"))
  plt.close()

plot_class_accuracy_bar(arg_max_preds, loader.test_y, loader.num_classes)

# print(np.mean([[True,False], [True, True], [False, False]], axis = 1))

# sns.set_style("whitegrid")
# pred_scatter = []
# truth_scatter = []
# for i in range(0, all_preds.shape[0], 10):
#   for j in range(all_preds.shape[1]):
#     for k in range(all_preds.shape[2]):
#       if k == int(loader.test_y[i][j]):
#         pred_scatter.append(all_preds[i][j][k])
#         truth_scatter.append(k) 
# plt.figure(figsize=(30, 30))
# plt.scatter(pred_scatter, truth_scatter, s=1)
# plt.savefig(os.path.join(args.logdir, descriptor, "pred_truth_scatter.pdf"))
# plt.savefig(os.path.join(args.logdir, descriptor, "pred_truth_scatter.png"))
# plt.close()

# sns.set_style("whitegrid")
# pred_scatter = []
# truth_scatter = []
# for i in range(all_preds.shape[0]):
#   for j in range(all_preds.shape[1]):
#     for k in range(all_preds.shape[2]):
#       pred_scatter.append(all_preds[i][j][k])
#       if k == int(loader.test_y[i][j]):
#         truth_scatter.append(1) 
#       else:
#         truth_scatter.append(0)
# plt.figure(figsize=(100,100))
# plt.scatter(pred_scatter, truth_scatter, s=1)
# plt.savefig(os.path.join(args.logdir, descriptor, "pred_truth_scatter.pdf"))
# plt.savefig(os.path.join(args.logdir, descriptor, "pred_truth_scatter.png"))
# plt.close()

###HOW TO RUN THIS FILE: python evaluate.py --path logs/LSTM_nonadversarial_nonadversarial_regular/model.ckpt###