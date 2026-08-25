"""A minimal neural-network decoder for Stim circuits.

The network learns the direct mapping:

    detector events -> logical-observable flips

No matching decoder, detector error model, or hand-written correction stage is
used. Train one model for each circuit topology (for example, each distance).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import stim
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DirectMLPDecoder(nn.Module):
    """One-hidden-layer feed-forward neural decoder."""

    def __init__(self, num_detectors: int, num_observables: int, hidden_nodes: int = 64):
        super().__init__()
        self.num_observables = num_observables
        self.network = nn.Sequential(
            nn.Linear(num_detectors, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1 << num_observables),
        )

    def forward(self, detector_events: torch.Tensor) -> torch.Tensor:
        return self.network(detector_events)


def _bits_to_class(bits: np.ndarray) -> np.ndarray:
    """Converts logical bits into one class label per shot."""
    weights = 1 << np.arange(bits.shape[1], dtype=np.int64)
    return bits.astype(np.int64) @ weights


def train_decoder(
    circuit: stim.Circuit,
    *,
    training_shots: int = 50_000,
    hidden_nodes: int = 64,
    epochs: int = 10,
    batch_size: int = 1024,
) -> DirectMLPDecoder:
    """Trains directly on Stim's observable-flip samples."""
    detector_events, observable_flips = circuit.compile_detector_sampler().sample(
        training_shots, separate_observables=True
    )
    features = torch.tensor(detector_events, dtype=torch.float32)
    labels = torch.tensor(_bits_to_class(observable_flips), dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)

    model = DirectMLPDecoder(
        circuit.num_detectors, circuit.num_observables, hidden_nodes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def decode_batch(model: DirectMLPDecoder, detector_events: np.ndarray) -> np.ndarray:
    """Predicts logical-observable flips directly from detector events."""
    classes = model(torch.tensor(detector_events, dtype=torch.float32)).argmax(dim=1)
    return np.stack(
        [(classes.numpy() >> bit) & 1 for bit in range(model.num_observables)], axis=1
    ).astype(np.uint8)


def count_logical_errors(
    circuit: stim.Circuit, model: DirectMLPDecoder, num_shots: int
) -> int:
    detector_events, actual_logical = circuit.compile_detector_sampler().sample(
        num_shots, separate_observables=True
    )
    predicted_logical = decode_batch(model, detector_events)
    return np.count_nonzero(np.any(actual_logical != predicted_logical, axis=1))


if __name__ == "__main__":
    num_shots = 10_000
    training_noise = 0.01
    test_noises = [0.001, 0.003, 0.005, 0.01, 0.02]

    for distance in [3, 5, 7]:
        training_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds=distance * 3,
            distance=distance,
            before_round_data_depolarization=training_noise,
        )
        model = train_decoder(training_circuit)

        logical_error_rates = []
        for noise in test_noises:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=distance * 3,
                distance=distance,
                before_round_data_depolarization=noise,
            )
            errors = count_logical_errors(circuit, model, num_shots)
            logical_error_rates.append(errors / num_shots)
        plt.plot(test_noises, logical_error_rates, label=f"d={distance}")

    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()
