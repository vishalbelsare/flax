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

"""Tests for flax.examples.ogbg_molpcba.input_pipeline."""

from absl.testing import absltest
import input_pipeline
import jax.numpy as jnp
import jraph


class DataLoaderTest(absltest.TestCase):

  def test_nearest_power_of_two(self):
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(1), 2)
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(2), 4)
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(3), 4)
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(4), 8)
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(500), 512)
    self.assertEqual(input_pipeline._nearest_bigger_power_of_two(513), 1024)

  def test_graphs_tuple_padding(self):
    # Dummy graph.
    graphs = jraph.GraphsTuple(
        n_node=jnp.arange(3, 11),
        n_edge=jnp.arange(4, 12),
        senders=jnp.zeros(60, dtype=jnp.int32),
        receivers=jnp.ones(60, dtype=jnp.int32),
        nodes=jnp.zeros((52, 10)),
        edges=jnp.zeros((60, 10)),
        globals=jnp.zeros((8, 10)),
    )
    self.assertLen(graphs.nodes, 52)
    self.assertLen(graphs.edges, 60)
    self.assertLen(graphs.n_node, 8)

    # We can still pad!
    padded_graphs = input_pipeline.pad_to_nearest_power_of_two(graphs,
                                                               batch_size=8)
    self.assertLen(padded_graphs.nodes, 64)
    self.assertLen(padded_graphs.edges, 64)
    self.assertLen(padded_graphs.n_node, 9)

  def test_get_raw_datasets(self):
    datasets = input_pipeline.get_raw_datasets()
    self.assertIn('train', datasets)
    self.assertIn('validation', datasets)
    self.assertIn('test', datasets)

    self.assertLen(datasets['train'], 350343)
    self.assertLen(datasets['validation'], 43793)
    self.assertLen(datasets['test'], 43793)

    # Test that we can loop over each of these datasets.
    for split in ['train', 'validation', 'test']:
      for _ in datasets[split]:
        break

  def test_get_datasets_as_graphs_tuples(self):
    datasets = input_pipeline.get_datasets_as_graphs_tuples()
    self.assertIn('train', datasets)
    self.assertIn('validation', datasets)
    self.assertIn('test', datasets)

    # Test that we can loop over each of these datasets.
    for split in ['train', 'validation', 'test']:
      for _ in datasets[split]:
        break

  def test_batching_valid(self):
    for valid_batch_size in [1, 5, 12, 100]:
      datasets = input_pipeline.get_datasets_as_graphs_tuples(
          batch_size=valid_batch_size, pad=False)

      for dataset in datasets.values():
        batch = next(dataset)
        self.assertLen(batch.n_node, valid_batch_size)
        self.assertLen(batch.globals, valid_batch_size)

      datasets = input_pipeline.get_datasets_as_graphs_tuples(
          batch_size=valid_batch_size, pad=True)
      for dataset in datasets.values():
        batch = next(dataset)
        self.assertLen(batch.n_node, valid_batch_size + 1)
        self.assertLen(batch.globals, valid_batch_size + 1)

  def test_batching_invalid(self):
    for invalid_batch_size in [-1, 0]:

      with self.assertRaises(ValueError):
        datasets = input_pipeline.get_datasets_as_graphs_tuples(
            batch_size=invalid_batch_size, pad=False)

        # We need this because input_pipeline.convert_to_graphs_tuple()
        # will only run if looped over.
        for _ in datasets['train']:
          break

      with self.assertRaises(ValueError):
        datasets = input_pipeline.get_datasets_as_graphs_tuples(
            batch_size=invalid_batch_size, pad=True)

        # We need this because input_pipeline.convert_to_graphs_tuple()
        # will only run if looped over.
        for _ in datasets['train']:
          break


if __name__ == '__main__':
  absltest.main()
