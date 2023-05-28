import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

folder_path = "logs/accuracies"

models = []
model_accuracies = []
# Iterate through files in the folder
for filename in os.listdir(folder_path):
  file_path = os.path.join(folder_path, filename)
  models.append(filename[:-4])
  # Check if the path is a file
  if os.path.isfile(file_path):
    # Open the file for reading
    with open(file_path, "r") as file:
      accuracy = file.read()
      model_accuracies.append(float(accuracy))

sns.set_style("whitegrid")
def plot_all_models_accuracy_bar(models, model_accuracies):
  x = np.arange(len(models))
  plt.bar(models, model_accuracies)
  plt.xlabel('Model')
  plt.ylabel('Accuracy')
  plt.title("Model Prediction Accuracies")
  plt.xticks(rotation=60, ha = 'right')
  plt.tight_layout()
  plt.savefig(os.path.join("logs", "model_accuracies.pdf"))
  plt.savefig(os.path.join("logs", "model_accuracies.png"))
  plt.close()

plot_all_models_accuracy_bar(models, model_accuracies)