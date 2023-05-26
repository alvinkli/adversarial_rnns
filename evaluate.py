import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

import data
import models

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, help="Path to the saved model.ckpt")

parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--num_nodes', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
total_loss = loss_fn(loader.test_y, all_preds)
total_accuracy = np.mean(loader.test_y == np.argmax(all_preds, -1))

print(f"total_loss: {total_loss} \t total_accuracy: {total_accuracy}")

### Plot any additional plots here -- all variables should be computed and
# ready to visualize: (ground truth = loader.test_y) and predicted (logits) = all_preds
import pdb; pdb.set_trace()
