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

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the path is a file
    if os.path.isfile(file_path):
        # Open the file for reading
        with open(file_path, "r") as file:
            # Read the contents of the file
            file_contents = file.read()
            
            # Process the file contents
            # ... (your code here)
            
            # Print the contents of the file
            print(file_contents)