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

"""Defines a sweep for the hyperparameters for the GNN."""

import ml_collections


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep('config.batch_size', [1, 2, 5, 10, 20]),
      hyper.sweep('config.message_passing_steps', [1, 2, 3, 4, 5]),
      hyper.sweep('config.num_training_epochs', [1, 5, 10, 20]),
      hyper.sweep('config.latent_size', [10, 20, 50, 100]),
  ])


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Optimizer.
  config.optimizer = 'adam'
  config.learning_rate = 1e-4

  # Training hyperparameters.
  config.batch_size = None
  config.num_training_epochs = None

  # GNN hyperparameters.
  config.message_passing_steps = None
  config.latent_size = None
  config.mlp_feature_sizes = (50, 50)

  return config
