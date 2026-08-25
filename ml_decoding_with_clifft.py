"""Minimal loss-aware neural decoder using Clifft's neutral-atom noise model.

The network is trained directly on information a decoder could receive:

    [surface-code detector events | loss-herald bits] -> logical observable flip

``final_status`` is deliberately *not* used: it is simulator-only ground truth,
not information a real decoder receives.

Install the tutorial requirements first, then run this file.  Clifft's
noncomputational sampler supplies the leakage/loss trajectories, detector events,
loss heralds, and logical observables; PyTorch supplies the small MLP.
"""

from __future__ import annotations

import numpy as np
import stim
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import clifft
    from clifft import noncomp
except ImportError as error:
    raise SystemExit("Install dependencies with `pip install -r requirements.txt`.") from error


Level = noncomp.Level


def transition_matrix(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    """Build Clifft's T[destination][source] five-level transition matrix."""
    matrix = [[0.0] * 5 for _ in range(5)]
    for (destination, source), probability in entries.items():
        matrix[destination][source] = probability
    return matrix


def loss_classifier() -> noncomp.Classifier:
    """Read 0/1 normally and herald an atom-loss measurement outcome."""
    matrix = [[0.0] * 5 for _ in range(3)]
    matrix[0][Level.G] = matrix[1][Level.E] = 1.0
    matrix[0][Level.LEAK_G] = 1.0
    matrix[1][Level.LEAK_E] = 1.0
    matrix[2][Level.LOST] = 1.0
    # Clifft 0.7 encodes the two binary outcomes and optional third herald row
    # directly in the matrix; it does not take separate symbol names.
    return noncomp.Classifier(matrix)


def neutral_atom_model(noise_scale: float = 1.0) -> noncomp.Model:
    """A small gate-dependent leakage/loss model adapted from Clifft's example."""
    leak_2q = 1.0e-3 * noise_scale
    loss_2q = 3.9e-3 * noise_scale
    leak_1q = 1.3e-3 * noise_scale
    return noncomp.Model(
        initial_state=[1 - 0.0095 * noise_scale, 0, 0.0065 * noise_scale, 0.003 * noise_scale, 0],
        transitions={
            "CX": transition_matrix(
                {
                    (Level.LEAK_E, Level.E): leak_2q,
                    (Level.LOST, Level.E): loss_2q,
                    (Level.LEAK_G, Level.G): 1.0e-4 * noise_scale,
                }
            ),
            "S": transition_matrix({(Level.LEAK_E, Level.E): leak_1q}),
        },
        classifier=loss_classifier(),
        reset_restores_lost=True,
        # Clifft 0.7 represents the documented state-dependent-transition
        # approximation as ``damping="neglect"``. Newer Clifft releases call
        # the corresponding policy ``equalize_rates``.
        damping="neglect",
    )


def make_surface_code(distance: int = 3, rounds: int = 3) -> stim.Circuit:
    """A rotated surface-code memory circuit with one logical observable."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=distance, rounds=rounds
    )


def sample_features_and_labels(
    circuit: stim.Circuit, model: noncomp.Model, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Samples decoder inputs and labels from Clifft.

    The circuit's OBSERVABLE_INCLUDE annotation produces ``result.observables``.
    ``result.heralds`` contains the physically available loss flags at each
    measurement slot, while ``result.final_status`` remains intentionally hidden
    from the decoder.
    """
    result = noncomp.sample(str(circuit), model, shots=shots, seed=seed)
    detector_events = np.asarray(result.detectors, dtype=np.float32)
    loss_heralds = np.asarray(result.heralds, dtype=np.float32)
    observable_flips = np.asarray(result.observables, dtype=np.uint8)
    features = np.concatenate([detector_events, loss_heralds], axis=1)
    return features, observable_flips


def bits_to_classes(bits: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(bits.shape[1], dtype=np.int64)
    return bits.astype(np.int64) @ weights


class LossAwareMLP(nn.Module):
    """A one-hidden-layer MLP: detector bits and loss flags to logical class."""

    def __init__(self, input_size: int, num_observables: int, hidden_size: int = 64):
        super().__init__()
        self.num_observables = num_observables
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1 << num_observables),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def train_decoder(
    features: np.ndarray,
    observable_flips: np.ndarray,
    *,
    hidden_size: int = 64,
    epochs: int = 10,
    batch_size: int = 1024,
) -> LossAwareMLP:
    inputs = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(bits_to_classes(observable_flips), dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=True)
    decoder = LossAwareMLP(features.shape[1], observable_flips.shape[1], hidden_size)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    decoder.train()
    for _ in range(epochs):
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(decoder(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
    return decoder.eval()


@torch.no_grad()
def predict(decoder: LossAwareMLP, features: np.ndarray) -> np.ndarray:
    classes = decoder(torch.tensor(features, dtype=torch.float32)).argmax(dim=1).numpy()
    return np.stack(
        [(classes >> bit) & 1 for bit in range(decoder.num_observables)], axis=1
    ).astype(np.uint8)


if __name__ == "__main__":
    circuit = make_surface_code(distance=3, rounds=3)
    train_features, train_labels = sample_features_and_labels(
        circuit, neutral_atom_model(), shots=20_000, seed=1
    )
    decoder = train_decoder(train_features, train_labels)

    test_features, test_labels = sample_features_and_labels(
        circuit, neutral_atom_model(), shots=10_000, seed=2
    )
    logical_error_rate = np.mean(np.any(predict(decoder, test_features) != test_labels, axis=1))
    print(f"NN input shape (detectors + loss heralds): {train_features.shape}")
    print(f"Logical error rate: {logical_error_rate:.3%}")
