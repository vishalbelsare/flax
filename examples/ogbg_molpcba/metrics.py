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

"""Metrics used for training and evaluation."""

from typing import Optional
import jax
import jax.numpy as jnp



@jax.jit
def binary_cross_entropy_with_mask(probs: jnp.ndarray,
                                   targets: jnp.ndarray,
                                   mask: jnp.ndarray):
  """Binary cross entropy loss for predictions within [0, 1], with masked elements."""
  # To prevent propagation of NaNs during grad().
  targets = jnp.where(mask, targets, 0.5)

  # For numerical stability.
  eps = 1e-7

  loss = (jnp.log(eps + probs) * targets) + \
         (jnp.log(1 + eps - probs) * (1 - targets))
  loss = jnp.where(mask, loss, 0)
  return -jnp.sum(loss) / jnp.sum(mask)


@jax.jit
def accuracy_with_mask(probs: jnp.ndarray,
                       targets: jnp.ndarray,
                       mask: jnp.ndarray):
  """Accuracy for predictions within [0, 1], with masked elements."""
  # Threshold to get predictions.
  preds = (probs > 0.5)
  labels_match_targets = (preds == targets)
  labels_match_targets = jnp.where(mask, labels_match_targets, False)
  return jnp.sum(labels_match_targets) / jnp.sum(mask)


@jax.jit
def precision_and_recall(preds: jnp.ndarray,
                         targets: jnp.ndarray) -> jnp.ndarray:
  """Returns the precision and recall for the given predictions."""
  # Compute true positives, false positives and false negatives.
  tps = jnp.sum((preds == 1) & (targets == 1), axis=0)
  fps = jnp.sum((preds == 1) & (targets == 0), axis=0)
  fns = jnp.sum((preds == 0) & (targets == 1), axis=0)

  # Then, compute task-wise precision and recall.
  precision = jnp.where((tps + fps) > 0, tps / (tps + fps), 0)
  recall = tps / (tps + fns)

  return precision, recall


@jax.jit
def average_precision(probs: jnp.ndarray,
                      targets: jnp.ndarray):
  """Average precision for multiple tasks over 100 thresholds uniformly spaced in [0, 1]."""
  def precision_and_recall_at_threshold(threshold):
    # Threshold probabilities.
    preds = (probs >= threshold)

    # Get task-wise precision and recall.
    precision, recall = precision_and_recall(preds, targets)

    return precision, recall

  thresholds = jnp.flip(jnp.linspace(0, 1, num=100))
  precisions, recalls = jax.lax.map(precision_and_recall_at_threshold,
                                    thresholds)
  recalls_diff = jnp.append(jnp.zeros((1, targets.shape[1])),
                            jnp.diff(recalls, axis=0), axis=0)
  stepwise_products = jnp.multiply(recalls_diff, precisions)
  return jnp.sum(stepwise_products, axis=0)
