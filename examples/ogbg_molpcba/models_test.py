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

"""Tests for flax.examples.ogbg_molpcba.models."""

from absl.testing import absltest
import models
import jax
import jax.numpy as jnp
import jraph


class ModelsTest(absltest.TestCase):

  def test_gnn(self):
    """Tests GNN model interface with dummy inputs."""
    rng = jax.random.PRNGKey(0)
    graphs = jraph.GraphsTuple(
        n_node=jnp.arange(3, 11),
        n_edge=jnp.arange(4, 12),
        senders=jnp.zeros(60, dtype=jnp.int32),
        receivers=jnp.ones(60, dtype=jnp.int32),
        nodes=jnp.zeros((52, 10)),
        edges=jnp.zeros((60, 10)),
        globals=jnp.zeros((8, 10)),
    )

    # First model.
    net = models.GNN(latent_size=4,
                     mlp_feature_sizes=[3, 5],
                     output_globals_size=100)
    output, _ = net.init_with_output(rng, graphs)

    # Output should be graph with the same topology, but a
    # different number of features.
    allclose = jnp.allclose
    self.assertIsInstance(output, jraph.GraphsTuple)
    self.assertTrue(allclose(output.n_node, graphs.n_node))
    self.assertTrue(allclose(output.n_edge, graphs.n_edge))
    self.assertTrue(allclose(output.senders, graphs.senders))
    self.assertTrue(allclose(output.receivers, graphs.receivers))
    self.assertEqual(output.nodes.shape, (52, 5))
    self.assertEqual(output.edges.shape, (60, 5))
    self.assertEqual(output.globals.shape, (8, 100))

    # Second model.
    net = models.GNN(latent_size=10,
                     mlp_feature_sizes=[3],
                     output_globals_size=150)
    output, _ = net.init_with_output(rng, graphs)

    # Output should be graph with the same topology, but a
    # different number of features.
    self.assertIsInstance(output, jraph.GraphsTuple)
    self.assertTrue(allclose(output.n_node, graphs.n_node))
    self.assertTrue(allclose(output.n_edge, graphs.n_edge))
    self.assertTrue(allclose(output.senders, graphs.senders))
    self.assertTrue(allclose(output.receivers, graphs.receivers))
    self.assertEqual(output.nodes.shape, (52, 3))
    self.assertEqual(output.edges.shape, (60, 3))
    self.assertEqual(output.globals.shape, (8, 150))

if __name__ == '__main__':
  absltest.main()
