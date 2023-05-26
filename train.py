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
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_nodes', type=int, default=128)
parser.add_argument('--num_iters', type=int, default=4000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--model_type', type=str, default="LSTM")
parser.add_argument('--train', type=str, default="nonadversarial")
parser.add_argument('--test', type=str, default="nonadversarial")
parser.add_argument('--sample', type=str, default="regular")
parser.add_argument('--logdir', type=str, default='./logs')
args = parser.parse_args()


# Setup dataloader and model
regular = args.sample == "regular"
loader = data.PersonData(args.seq_len, regular)
model = models.build_model(
  model_type=args.model_type,
  num_nodes=args.num_nodes,
  training_mode=args.train,
  testing_mode=args.test,
  batch_size=args.batch_size,
  seq_length=args.seq_len,
  num_x_features=loader.train_x.shape[2],
  regular=regular,
)
optimizer = tf.keras.optimizers.Adam(args.lr)


# Prepare for training, create the log directory and initialize loss histories
pbar = tqdm(range(args.num_iters))
train_losses = []
test_losses = []
dirname = f"{args.model_type}_{args.train}_{args.test}_{args.sample}"
Path(os.path.join(args.logdir, dirname)).mkdir(parents=True, exist_ok=True)
print(f"Saving to: {os.path.join(args.logdir, dirname)}")


# Loop through and train
for iter in pbar:

  # Get a train batch and train with it
  i = np.random.choice(loader.train_x.shape[0], args.batch_size)
  x, y, t = [tf.convert_to_tensor(arr) for arr in (loader.train_x[i], loader.train_y[i], loader.train_t[i])]
  input = (x, t) if ((not regular) and (args.model_type == "LTC")) else x
  train_loss = model.train_step(optimizer, input, y)
  train_losses.append(train_loss)

  # Every so often do the same for testing
  if iter % 20 == 0:
    iv = np.random.choice(loader.test_x.shape[0], args.batch_size)
    xv, yv, tv = [tf.convert_to_tensor(arr) for arr in (loader.test_x[iv], loader.test_y[iv], loader.test_t[iv])]
    inputv = (xv, tv) if ((not regular) and (args.model_type == "LTC")) else xv
    test_loss = model.test_step(inputv, yv)
    test_losses.append(test_loss)

  # Print the progress
  pbar.set_description(f"Loss {train_loss:.2f}   VLoss {test_loss:.2f}")

  # Save the progress
  if iter % 1000 == 0:
    model.save_weights(os.path.join(args.logdir, dirname, "model.ckpt"))

model.save_weights(os.path.join(args.logdir, dirname, "model.ckpt"))


# Make some preliminary (loss) plots, save them in the same directory as the model
plt.plot(np.linspace(0, args.num_iters, len(train_losses)), train_losses)
plt.plot(np.linspace(0, args.num_iters, len(test_losses)), test_losses)
plt.savefig(os.path.join(args.logdir, dirname, "losses.pdf"))
plt.savefig(os.path.join(args.logdir, dirname, "losses.png"))
plt.close()

# Next: run evaluate.py
