import tensorflow as tf
from ncps import wirings
from ncps.tf import CfC, LTC

def build_model(model_type, training_mode, testing_mode, batch_size, seq_length, num_x_features, regular, num_nodes=256, num_outputs=7, adv_eps=0.1):
  if model_type == "LSTM":
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(seq_length, num_x_features)),
      tf.keras.layers.LSTM(
        num_nodes,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=False,
      ),
      tf.keras.layers.Dense(num_outputs)
    ])
  elif model_type == "LTC":
    motor_neurons = num_outputs
    wiring = wirings.AutoNCP(num_nodes, motor_neurons)
    if regular:
        model = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(seq_length, num_x_features)),
          LTC(wiring, return_sequences=True),
        ])
    else:
        input_x = tf.keras.Input(shape=(seq_length, num_x_features))
        input_t = tf.keras.Input(shape=(seq_length, 1))
        ltc_layer = LTC(wiring, return_sequences=True)
        x = ltc_layer((input_x, input_t))
        y = tf.keras.layers.Dense(num_outputs)(x)
        model = tf.keras.Model(inputs=(input_x, input_t), outputs=y)
  return AdaptableRNNModel(model, model_type, training_mode, testing_mode, adv_eps)



class AdaptableRNNModel(tf.keras.Model):
  def __init__(self, model, model_type, training_mode, testing_mode, adversarial_eps=0.1):
    super(AdaptableRNNModel, self).__init__()
    self.model = model
    self.model_type = model_type
    self.training_mode = training_mode
    self.testing_mode = testing_mode
    self.adversarial_eps = adversarial_eps
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if self.adversarial_eps > 0:
        attack_iters = int(max(min(self.adversarial_eps, 4), 1))
        self.fgsm = IFGSM(self.adversarial_eps, attack_iters)

  def get_model_type(self):
    return self.model_type

  def get_testing_mode(self):
    return self.testing_mode

  def call(self, x):
    return self.model(x)

  def adv_call(self, input, y):
    attack_iters = int(max(min(self.adversarial_eps, 4), 1))
    fgsm = IFGSM(self.adversarial_eps, attack_iters)
    x = fgsm(self.model, self.compute_loss, x, y)
    return self.model(x)

  def compute_loss(self, y, pred):
    return self.loss_fn(y, pred)

  @tf.function
  def train_step(self, optimizer, input, y):
    if type(input) == tuple:
      x, t = input
    else:
      x, t = (input, None)

    if self.training_mode == "adversarial":
      if self.adversarial_eps > 0:
        x = self.fgsm(self.model, self.compute_loss, input, y)

    with tf.GradientTape() as tape:
      if type(input) == tuple:
        pred = self.call((x, t))
      else:
        pred = self.call(x)

      loss = self.compute_loss(y, pred)
    grads = tape.gradient(loss, self.model.trainable_variables)
    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return loss

  @tf.function
  def test_step(self, input, y):
    if type(input) == tuple:
      x, t = input
    else:
      x, t = (input, None)

    if self.testing_mode == "nonadversarial":
      if type(input) == tuple:
        pred = self.call((x, t))
      else:
        pred = self.call(x)

      loss = self.compute_loss(y, pred)
    else:
      attack_iters = int(max(min(self.adversarial_eps, 4), 1))
      fgsm = IFGSM(self.adversarial_eps, attack_iters)
      x = fgsm(self.model, self.compute_loss, input, y)

      if type(input) == tuple:
        y_hat = self.call((x, t))
      else:
        y_hat = self.call(x)

      loss = self.compute_loss(y, y_hat)
    return loss


class IFGSM:
  def __init__(self, epsilon, iterations=1):
    self.iterations = iterations
    self.epsilon = epsilon

  @tf.function
  def __call__(self, model, loss, input, y):
    if type(input) == tuple:
        x, t = input
    else:
        x, t = (input, None)

    for i in range(self.iterations):
      with tf.GradientTape() as tape:
        tape.watch(x)
        if type(input) == tuple:
          pred = model((x, t))
        else:
          pred = model(x)
        loss_value = tf.reduce_mean(loss(y, pred))
      gradient = tape.gradient(loss_value, x)
      gradient = tf.sign(gradient)
      gradient = gradient * (
        self.epsilon / tf.constant(self.iterations, dtype=tf.float32)
      )
      x = x + gradient
    return x
