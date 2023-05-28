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
  if regular:
    if test_mode == "adversarial":
      pred = model.adv_call(x)
    else:
      pred = model(x)
  else:
    if test_mode == "adversarial":
      pred = model.adv_call((x, t))
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
print(f"total_loss: {total_loss} \t total_accuracy: {total_accuracy}")

### Plot any additional plots here -- all variables should be computed and
# ready to visualize: (ground truth = loader.test_y) and predicted (logits) = all_preds
sns.set_style("whitegrid")

def plot_confusion_matrix(preds, labels, num_classes):
  all_preds= []
  all_labels = []
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      all_preds.append(preds[i][j])
      all_labels.append(labels[i][j])
  confusion_matrix = np.array(tf.math.confusion_matrix(all_labels, all_preds, num_classes))

  class_labels = ['State 0', 'State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6']

  # Plot the confusion matrix as a table
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.set_title(f"{descriptor} Matrix", x = 0.47, pad=0)
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

# print("HFIHSOIDHFIHOSHFIHSDOS", (loader.test_y == arg_max_preds).shape)
# print("\n\nHFIHSOIDHFIHOSHFIHSDOS", (loader.test_y == arg_max_preds)[0])

def plot_class_accuracy_matrix(preds, labels, num_classes):
  pred_correctness = (loader.test_y == arg_max_preds)
  class_pred_correctness = []
  for c in range(num_classes):
    class_pred_correctness.append([])
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      class_pred_correctness[int(labels[i][j])].append(pred_correctness[i][j])
  print(np.array(class_pred_correctness, dtype=object).shape)
  class_accuracies = []
  for i in range(num_classes):
    class_accuracies.append(np.mean(class_pred_correctness[i]))
  print(class_accuracies)

  x = np.arange(len(class_accuracies))
  plt.bar(x, class_accuracies)
  plt.xlabel('Class')
  plt.ylabel('Accuracy')
  plt.title(f"{descriptor} Prediction Accuracies")
  plt.savefig(os.path.join(args.logdir, descriptor, "class_accuracies.pdf"))
  plt.savefig(os.path.join(args.logdir, descriptor, "class_accuracies.png"))
  plt.close()
plot_class_accuracy_matrix(arg_max_preds, loader.test_y, loader.num_classes)

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