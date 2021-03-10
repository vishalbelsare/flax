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

"""Tests for flax.examples.ogbg_molpcba.train."""


import pathlib
import tempfile

from absl.testing import absltest
import flax
import input_pipeline
import train
from configs import default
from configs import test
import jax
from jax import numpy as jnp
import jraph
import tensorflow as tf
import tensorflow_datasets as tfds

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class OgbgMolpcbaTrainTest(absltest.TestCase):
  """Test cases for training on ogbg_molpcba."""

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Print the current platform (the default device).
    platform = jax.local_devices()[0].platform
    print('Running on platform:', platform.upper())

  def test_train_step(self):
    """Tests a single training step with the default config."""
    # Get the default configuration.
    config = default.get_config()

    # Initialize the network with a dummy graph.
    rng = jax.random.PRNGKey(0)
    dummy_graphs = input_pipeline.get_dummy_graphs()
    net = train.create_model(config)
    params = net().init(rng, dummy_graphs)

    # Create the optimizer.
    optimizer = train.create_optimizer(config, params)

    # Perform one step of updates.
    # We use the same batch of graphs that we used for initialization.
    optimizer = train.train_step(optimizer, net, dummy_graphs)

    # Check that none of the parameters are NaNs!
    params = flax.core.unfreeze(optimizer.target)
    flat_params = {
        '/'.join(k): v
        for k, v in flax.traverse_util.flatten_dict(params).items()
    }
    for array in flat_params.values():
      self.assertTrue(jnp.all(~jnp.isnan(array)))

  @staticmethod
  def get_dummy_dataset(dummy_graph):
    def as_dataset(self, *args, **kwargs):
      del args, kwargs
      dummy_graph_tf = {
          'labels': tf.convert_to_tensor(dummy_graph.globals[0]),
          'num_nodes': tf.convert_to_tensor(dummy_graph.n_node, dtype=tf.int32),
          'node_feat': tf.convert_to_tensor(dummy_graph.nodes),
          'edge_feat': tf.convert_to_tensor(dummy_graph.edges),
          'edge_index': tf.stack([dummy_graph.senders, dummy_graph.receivers],
                                 axis=1),
      }
      return tf.data.Dataset.from_generator(
          lambda: (dummy_graph_tf for _ in range(10)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )
    return as_dataset

  def test_train_and_evaluate(self):
    """Tests training and evaluation code."""
    # Create a temporary directory where tensorboard metrics are written.
    workdir = tempfile.mkdtemp()

    # # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    # Get the test configuration.
    config = test.get_config()

    # We need to specify how the mock data should look like,
    # since they must be valid graphs!
    # Here, we can just reuse one of the dummy graphs.
    with tfds.testing.mock_data():
      dummy_graphs = input_pipeline.get_dummy_graphs()
      dummy_graph = jraph.unbatch(dummy_graphs)[0]

    # Ensure train_and_evaluate() runs without any errors!
    with tfds.testing.mock_data(
        as_dataset_fn=self.get_dummy_dataset(dummy_graph)
    ):
      train.train_and_evaluate(config=config, workdir=workdir)

if __name__ == '__main__':
  absltest.main()
