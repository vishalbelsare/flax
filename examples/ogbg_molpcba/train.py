# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library file which executes the training and evaluation loop for ogbg-molpcba."""

# import functools
import logging
from typing import Dict, Iterator, Tuple, Callable

from flax import optim
import input_pipeline
import metrics
import models
import flax.linen as nn
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import jraph
import ml_collections


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The trained optimizer.
  """
  # Create writer for logs.
  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  # Initialize the network with a dummy graph.
  logging.info('Initializing network.')
  rng = jax.random.PRNGKey(0)
  dummy_graphs = input_pipeline.get_dummy_graphs()
  net = create_model(config)
  params = net().init(rng, dummy_graphs)

  # Create the optimizer, and transfer it to the default device.
  optimizer = create_optimizer(config, params)
  optimizer = jax.device_put(optimizer)

  # Begin training over epochs.
  optimizer = train_model(optimizer, net, config, summary_writer)
  return optimizer


def train_model(optimizer, net,
                config: ml_collections.ConfigDict,
                summary_writer):
  """Trains a model over the 'train' split, and reports statistics over all splits."""

  for epoch in range(config.num_training_epochs):
    # Get datasets, organized by split.
    datasets = input_pipeline.get_datasets_as_graphs_tuples(config.batch_size,
                                                            pad=True)

    # Train over one epoch of the training set.
    optimizer = train_epoch(optimizer, net, datasets['train'])

    # We log the loss and accuracy over all splits.
    evaluate_model(optimizer, net, epoch, summary_writer,
                   compute_mean_ap=False)

  logging.info('Training finished!')

  # At the end of training, we log the loss, accuracy and mean average precision
  # over all splits.
  evaluate_model(optimizer, net, config.num_training_epochs, summary_writer,
                 compute_mean_ap=True)

  return optimizer


def evaluate_model(optimizer, net, epoch, summary_writer,
                   compute_mean_ap=False):
  """Evaluates the model on metrics over all splits, called after training is complete."""
  # Since we compute statistics over the whole dataset,
  # we set a batch_size of None.
  datasets = input_pipeline.get_datasets_as_graphs_tuples(None,
                                                          pad=True)

  # Compute metrics.
  logging.info('Computing metrics!')
  all_metrics = {}
  for split in ['train', 'validation', 'test']:
    params = optimizer.target
    graphs = next(datasets[split])
    loss, aux = compute_metrics(params, net, graphs, compute_mean_ap)

    all_metrics[split] = {}
    all_metrics[split]['loss'] = loss
    all_metrics[split].update(aux)

  # Log statistics.
  for split, metrics_split in all_metrics.items():
    if compute_mean_ap:
      logging.info('epoch: %d, %s loss: %.4f, %s accuracy: %.2f, %s mAP: %.2f',
                   epoch,
                   split, metrics_split['loss'],
                   split, metrics_split['accuracy'] * 100,
                   split, metrics_split['average_precision'] * 100)
    else:
      logging.info('epoch: %d, %s loss: %.4f, %s accuracy: %.2f',
                   epoch,
                   split, metrics_split['loss'],
                   split, metrics_split['accuracy'] * 100)

    summary_writer.scalar('%s_loss' % split,
                          metrics_split['loss'],
                          epoch)
    summary_writer.scalar('%s_accuracy' % split,
                          metrics_split['accuracy'],
                          epoch)

    if compute_mean_ap:
      summary_writer.scalar('%s_average_precision' % split,
                            metrics_split['average_precision'],
                            epoch)

  summary_writer.flush()

  return optimizer


def train_epoch(optimizer, net,
                dataset: Iterator[jraph.GraphsTuple]):
  """Updates a model's parameters over one epoch of a dataset."""
  for graphs in dataset:
    optimizer = train_step(optimizer, net, graphs)
  return optimizer


@jax.partial(jax.jit, static_argnums=[1])
def train_step(optimizer, net,
               graphs: jraph.GraphsTuple):
  """Performs one update step over the current batch of graphs."""

  # The loss function.
  def loss_fn(params):
    return compute_metrics(params, net, graphs, False)

  # We don't use the value of the loss function, so we could use jax.grad()
  # instead of jax.value_and_grad() here, but we do so for consistency
  # with other Flax examples.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  # optimizer.target contains the current parameters of the model.
  (_, _), grad = grad_fn(optimizer.target)
  new_optimizer = optimizer.apply_gradient(grad)
  return new_optimizer


def create_model(config: ml_collections.ConfigDict) -> Callable[[], nn.Module]:
  return lambda: models.GNN(config.latent_size, config.mlp_feature_sizes,
                            config.message_passing_steps, 128)


def create_optimizer(config: ml_collections.ConfigDict, params):
  """Creates an optimizer according to the config, and assigns these parameters to the optimizer's target."""
  if config.optimizer == 'adam':
    opt = optim.Adam(learning_rate=config.learning_rate)
  elif config.optimizer == 'sgd':
    opt = optim.Momentum(learning_rate=config.learning_rate,
                         beta=config.momentum)
  else:
    raise ValueError('Unsupported optimizer.')

  return opt.create(params)


def get_predicted_probs(params, net,
                        graphs: jraph.GraphsTuple) -> jnp.ndarray:
  """Get predicted probabilties from the network for a given input GraphsTuple."""
  pred_graphs = net().apply(params, graphs)
  logits = pred_graphs.globals
  probs = jax.nn.sigmoid(logits)
  return probs


def get_mask(targets: jnp.ndarray, graphs: jraph.GraphsTuple) -> jnp.ndarray:
  """Gets the binary mask ignoring invalid targets and graphs."""
  # We have to ignore all NaN values - which indicate targets for which
  # the current graphs have no label.
  targets_mask = ~jnp.isnan(targets)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  graph_mask = jraph.get_graph_padding_mask(graphs)

  # Combine this with the mask over targets.
  mask = targets_mask & graph_mask[:, None]
  return mask


@jax.partial(jax.jit, static_argnums=[1, 3])
def compute_metrics(
        params,
        net,
        graphs: jraph.GraphsTuple,
        compute_mean_ap: bool = True
    ) -> Tuple[jnp.float32, Dict[str, jnp.ndarray]]:
  """Computes metrics over a set of graphs."""

  # The target labels our model has to predict.
  targets = graphs.globals

  # Get predictions.
  probs = get_predicted_probs(params, net, graphs)

  # Get the mask for invalid entries.
  mask = get_mask(targets, graphs)

  # Compute the loss.
  loss = metrics.binary_cross_entropy_with_mask(probs, targets, mask)

  # Compute accuracy.
  mean_accuracy = metrics.accuracy_with_mask(probs, targets, mask)
  aux = {
      'accuracy': mean_accuracy,
  }

  # Should we compute the average precision?
  # This is skipped during training, since it takes longer to compute.
  if compute_mean_ap:
    mean_average_precision = jnp.mean(metrics.average_precision(probs, targets))
    aux['average_precision'] = mean_average_precision

  return loss, aux


