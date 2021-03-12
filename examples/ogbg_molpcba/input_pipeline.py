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

"""Loads the ogbg-molpcba dataset from TensorFlow Datasets (tfds)."""

from typing import Dict, Sequence, Optional, Iterator

import jax.numpy as jnp
import jraph
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset_builder() -> tfds.core.DatasetBuilder:
  ds_builder = tfds.builder('ogbg_molpcba')
  return ds_builder


def get_raw_datasets() -> Dict[str, tf.data.Dataset]:
  """Returns datasets as tf.data.Dataset, organized by split."""
  ds_builder = get_dataset_builder()
  ds_builder.download_and_prepare()
  datasets = ds_builder.as_dataset()
  return datasets


def get_datasets_as_graphs_tuples(
    batch_size: Optional[int] = None,
    pad: bool = True) -> Dict[str, Iterator[jraph.GraphsTuple]]:
  """Returns datasets as GraphsTuples, organized by split."""
  datasets = get_raw_datasets()

  # Convert to numpy arrays.
  datasets_as_numpy = {
      split: tfds.as_numpy(dataset_split)
      for split, dataset_split in datasets.items()
  }

  # And then to GraphsTuple.
  # We can't use tf.data.Dataset's batch() method because the graphs
  # are of different sizes, so we batch with jraph.batch().
  datasets_as_graphs_tuple = {
      split: convert_to_graphs_tuple(dataset_numpy, batch_size, pad)
      for split, dataset_numpy in datasets_as_numpy.items()
  }
  return datasets_as_graphs_tuple


def convert_to_graphs_tuple(
    dataset: Sequence[Dict[str, np.ndarray]],
    batch_size: Optional[int],
    pad: bool) -> Iterator[jraph.GraphsTuple]:
  """Converts a dataset of edges, nodes and features to batches of GraphsTuples."""
  def convert_single_graph(graph: Dict[str, np.ndarray]) -> jraph.GraphsTuple:
    return jraph.GraphsTuple(
        n_node=jnp.asarray(graph['num_nodes']),
        n_edge=jnp.asarray([graph['edge_feat'].shape[0]]),
        nodes=jnp.asarray(graph['node_feat']),
        edges=jnp.asarray(graph['edge_feat']),
        senders=jnp.asarray(graph['edge_index'][:, 0]),
        receivers=jnp.asarray(graph['edge_index'][:, 1]),
        globals=jnp.asarray([graph['labels']]),
    )

  # Invalid batch size.
  if batch_size is not None and batch_size <= 0:
    raise ValueError('Batch size must be >= 0.')

  # No batching if 'batch_size' is None.
  elif batch_size is None:
    try:
      batch_size = len(dataset)
    except TypeError:
      batch_size = sum(1 for _ in dataset)

  current_batch = []
  for index, graph in enumerate(dataset):
    # Add to the current batch.
    current_batch.append(convert_single_graph(graph))

    # Is this batch full?
    if index % batch_size == (batch_size - 1):
      batched = jraph.batch(current_batch)

      # Add an extra graph to the current batch, so that the number of nodes
      # and number of edges are exactly powers of two.
      # This helps because JAX recompiles jitted functions whenever it is called
      # with inputs with new shapes. By padding, there are far fewer unique
      # input shapes that the jitted functions see later.
      if pad:
        batched = pad_to_nearest_power_of_two(batched, batch_size)

      yield batched
      current_batch = []


def _nearest_bigger_power_of_two(x: int) -> int:
  """Returns the nearest bigger power of 2.

  For example,
    1 -> 2
    2 -> 4
    3 -> 4
  and so on.

  Idea taken from:
  https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

  Args:
    x: an integer, generally representing the size of some array here.
  """
  y = 2
  while y <= x:
    y *= 2
  return y


def pad_to_nearest_power_of_two(batch: jraph.GraphsTuple,
                                batch_size: int) -> jraph.GraphsTuple:
  """Pads a batched GraphsTuple with extra graphs.

  Idea taken from:
  https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

  For example, if a GraphsTuple has:
      16 nodes,
      25 edges, and,
      5 graphs,
  we add an extra graph so that the resulting GraphsTuple has
      32 nodes,
      32 edges, and,
      5 + 1 = 6 graphs.

  Args:
    batch: Any GraphsTuple with a batch of graphs. Can have only one graph too!
    batch_size: The common batch size, since a batch can sometimes have
      fewer graphs.

  Returns:
    padded_batch: A padded GraphsTuple, with an extra graph such that
      the total number of edges and nodes is a power of two.
  """
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(batch.n_node))
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(batch.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = batch_size + 1

  return jraph.pad_with_graphs(batch,
                               pad_nodes_to, pad_edges_to, pad_graphs_to)


def get_dummy_graphs() -> jraph.GraphsTuple:
  """Returns a dummy GraphsTuple for initialization."""
  return jraph.GraphsTuple(
      n_node=jnp.arange(3, 11),
      n_edge=jnp.arange(4, 12),
      senders=jnp.zeros(60, dtype=jnp.int32),
      receivers=jnp.ones(60, dtype=jnp.int32),
      nodes=jnp.zeros((52, 9)),
      edges=jnp.zeros((60, 3)),
      globals=jnp.zeros((8, 128)),
  )
