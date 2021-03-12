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

"""Definition of the GNN model."""

# See issue #620.
# pytype: disable=wrong-arg-types

from typing import Sequence, Callable

from flax import linen as nn
import jax.numpy as jnp
import jraph


class MLP(nn.Module):
  """A MLP defined with Flax primitives."""

  feature_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, layer in enumerate([nn.Dense(size) for size in self.feature_sizes]):
      x = layer(x)
      if i != len(self.feature_sizes) - 1:
        x = self.activation(x)
    return x


class GNN(nn.Module):
  """A Graph Network model defined with Flax and Jraph."""

  latent_size: int
  mlp_feature_sizes: Sequence[int]
  message_passing_steps: int
  output_globals_size: int

  @nn.compact
  def __call__(self, graphs):
    # Replace the global parameter for graph classification.
    # We will readout this parameter as the logits.
    graphs = graphs._replace(globals=jnp.zeros([graphs.n_node.shape[0], 1]))

    # We will first linearly project the original features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.latent_size),
        embed_edge_fn=nn.Dense(self.latent_size),
        embed_global_fn=nn.Dense(self.latent_size))

    processed_graphs = embedder(graphs)
    for _ in range(self.message_passing_steps):
      # Now, apply a Graph Network for each message-passing round!
      net = jraph.GraphNetwork(
          update_node_fn=jraph.concatenated_args(
              MLP(self.mlp_feature_sizes)),
          # update_edge_fn=jraph.concatenated_args(
          #     MLP(self.mlp_feature_sizes)),
          update_edge_fn=None,
          update_global_fn=jraph.concatenated_args(
              MLP(self.mlp_feature_sizes)))
      processed_graphs = net(processed_graphs)

    # Since our graph-level predictions will be at globals, we now
    # decode to get the required output.
    decoder = jraph.GraphMapFeatures(
        embed_global_fn=nn.Dense(self.output_globals_size))
    return decoder(processed_graphs)
