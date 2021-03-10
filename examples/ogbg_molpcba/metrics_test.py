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

"""Tests for flax.examples.ogbg_molpcba.metrics."""


from absl.testing import absltest
import metrics
import jax
from jax import numpy as jnp


# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class OgbgMolpcbaMetricsTest(absltest.TestCase):
  """Tests for the various metrics in the ogbg_molpcba example."""

  def test_binary_cross_entropy_loss(self):
    """Tests for the loss computation."""

    probs = jnp.asarray([0.8, 0.9, 0.3, 1])
    targets = jnp.asarray([1, 0, 1, jnp.nan])
    mask = ~jnp.isnan(targets)

    loss = metrics.binary_cross_entropy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(loss,
                           -jnp.mean(jnp.log(jnp.asarray([0.8, 0.1, 0.3]))),
                           places=5)

    mask = jnp.asarray([True, True, False, False])
    loss = metrics.binary_cross_entropy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(loss,
                           -jnp.mean(jnp.log(jnp.asarray([0.8, 0.1]))),
                           places=5)

    mask = jnp.asarray([True, False, False, False])
    loss = metrics.binary_cross_entropy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(loss,
                           -jnp.mean(jnp.log(jnp.asarray([0.8]))),
                           places=5)

  def test_accuracy(self):
    """Tests for the accuracy computation."""
    probs = jnp.asarray([0.8, 0.9, 0.3, 1])
    targets = jnp.asarray([1, 0, 1, jnp.nan])
    mask = ~jnp.isnan(targets)

    accuracy = metrics.accuracy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(accuracy, 1/3)

    mask = jnp.asarray([True, True, False, False])
    accuracy = metrics.accuracy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(accuracy, 1/2)

    mask = jnp.asarray([True, False, False, False])
    accuracy = metrics.accuracy_with_mask(probs, targets, mask)
    self.assertAlmostEqual(accuracy, 1)

  def test_precision_recall(self):
    # Two task case.
    probs = jnp.asarray([[0.1, 0.2],
                         [0.2, 0.3],
                         [0.4, 0.5],
                         [0.6, 0.7]])
    targets = jnp.asarray([[0, 0],
                           [1, 1],
                           [0, 0],
                           [1, 1]])

    # Check at different thresholds.
    allclose = jnp.allclose
    preds = (probs >= 0.01)
    precision, recall = metrics.precision_and_recall(preds, targets)
    self.assertTrue(allclose(precision, [2/4, 2/4]))
    self.assertTrue(allclose(recall, [2/2, 2/2]))

    preds = (probs >= 0.15)
    precision, recall = metrics.precision_and_recall(preds, targets)
    self.assertTrue(allclose(precision, [2/3, 2/4]))
    self.assertTrue(allclose(recall, [2/2, 2/2]))

    preds = (probs >= 0.25)
    precision, recall = metrics.precision_and_recall(preds, targets)
    self.assertTrue(allclose(precision, [1/2, 2/3]))
    self.assertTrue(allclose(recall, [1/2, 2/2]))

    preds = (probs >= 0.55)
    precision, recall = metrics.precision_and_recall(preds, targets)
    self.assertTrue(allclose(precision, [1/1, 1/1]))
    self.assertTrue(allclose(recall, [1/2, 1/2]))

    preds = (probs >= 0.75)
    precision, recall = metrics.precision_and_recall(preds, targets)
    self.assertTrue(allclose(precision, [0, 0]))
    self.assertTrue(allclose(recall, [0, 0]))

  def test_average_precision(self):
    """Tests for the average precision computation."""
    # Two task case.
    probs = jnp.asarray([[0.1, 0.2],
                         [0.2, 0.3],
                         [0.4, 0.5],
                         [0.6, 0.7]])
    targets = jnp.asarray([[0, 0],
                           [1, 1],
                           [0, 0],
                           [1, 1]])

    # Compute average precision for each task.
    average_precision = metrics.average_precision(probs, targets)

    # Check for each task.
    self.assertTrue(jnp.allclose(average_precision, [2/3 * 1/2 + 1/1 * 1/2,
                                                     2/3 * 1/2 + 1/1 * 1/2]))
if __name__ == '__main__':
  absltest.main()
