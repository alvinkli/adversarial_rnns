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

regular_models = []
irregular_models = []
regular_model_accuracies = []
irregular_model_accuracies = []
# Iterate through files in the folder
for filename in os.listdir(folder_path):
  file_path = os.path.join(folder_path, filename)
  descriptor = filename
  model_type, train_mode, test_mode, sample = descriptor.split("_")
  formatted_model_name = f"Model Type: {model_type}\nTrain Mode: {train_mode}\nTest Mode: {test_mode}\nSample Frequency: {sample}"
  if sample == "regular":
    regular_models.append(formatted_model_name)
  else:
    irregular_models.append(formatted_model_name)
  # Check if the path is a file
  if os.path.isfile(file_path):
    # Open the file for reading
    with open(file_path, "r") as file:
      accuracy = file.read()
      if sample == "regular":
        regular_model_accuracies.append(float(accuracy))
      else:
        irregular_model_accuracies.append(float(accuracy))

def sort_models(model_names, model_accuracies):
  new_model_names = []
  new_model_accuracies = []

  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LSTM" and train_mode == "nonadversarial" and test_mode == "nonadversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LSTM" and train_mode == "nonadversarial" and test_mode == "adversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])

  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LSTM" and train_mode == "adversarial" and test_mode == "nonadversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LSTM" and train_mode == "adversarial" and test_mode == "adversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])

  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LTC" and train_mode == "nonadversarial" and test_mode == "nonadversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LTC" and train_mode == "nonadversarial" and test_mode == "adversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
    
  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LTC" and train_mode == "adversarial" and test_mode == "nonadversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
  for i in range(len(model_names)):
    name = model_names[i]
    model_type, train_mode, test_mode, sample = name.split("\n")
    model_type = model_type[12:]
    train_mode = train_mode[12:]
    test_mode = test_mode[11:]
    sample = sample[18:]
    if(model_type == "LTC" and train_mode == "adversarial" and test_mode == "adversarial"):
      new_model_names.append(name)
      new_model_accuracies.append(model_accuracies[i])
    
  return new_model_names, new_model_accuracies
    
regular_models_sorted, regular_model_accuracies_sorted = sort_models(regular_models, regular_model_accuracies)
irregular_models_sorted, irregular_model_accuracies_sorted = sort_models(irregular_models, irregular_model_accuracies)
# print(len(regular_models), len(irregular_model_accuracies))

sns.set_style("whitegrid")
def plot_all_models_accuracy_bar(models, model_accuracies, regular):
  x = np.arange(len(models))
  colors = ['steelblue', 'lightcoral', 'steelblue', 'lightcoral', 'steelblue', 'lightcoral', 'steelblue', 'lightcoral']
  plt.figure(figsize=(20, 15))
  plt.bar(models, model_accuracies, 1.0, color=colors)
  plt.xlabel('Model', fontsize = 20, labelpad = 15)
  plt.ylabel('Accuracy', fontsize = 20, labelpad = 15)
  if regular:
    plt.title("Regularly Sampled Model Prediction Accuracies", fontsize = 30, pad = 20.0)
  else:
    plt.title("Irregularly Sampled Model Prediction Accuracies", fontsize = 30, pad = 20.0)
  # plt.xticks(rotation=30, ha = 'right')
  plt.tight_layout()
  if regular:
    plt.savefig(os.path.join("logs", "regular_model_accuracies.pdf"))
    plt.savefig(os.path.join("logs", "regular_model_accuracies.png"))
  else:
    plt.savefig(os.path.join("logs", "irregular_model_accuracies.pdf"))
    plt.savefig(os.path.join("logs", "irregular_model_accuracies.png"))
  plt.close()

plot_all_models_accuracy_bar(regular_models_sorted, regular_model_accuracies_sorted, True)
plot_all_models_accuracy_bar(irregular_models_sorted, irregular_model_accuracies_sorted, False)