#!/usr/bin/env python3
"""Clifft loss-aware local predecoder with the existing Ising/PyMatching path.

This intentionally preserves the Ising-Decoding memory-circuit convention:

    4 usual syndrome/geometry channels + 1 loss-herald channel
        -> four existing local-predecoder heads
        -> existing residual-syndrome and local-frame conversion
        -> ordinary PyMatching global decoder

Clifft supplies gate-triggered leakage/loss dynamics.  The fifth channel is
the final data-readout loss herald, broadcast over the block's rounds.  It is
therefore an offline, post-block predecoder input, not a causal streaming
signal.  Leakage has no separate herald in this model, but its effect remains
in the sampled detector record.

The local targets are deliberately conservative: a correction is proposed only
when every visible adjacent check at a heralded loss location fires.  Ambiguous
loss events stay in the residual syndrome for PyMatching, which still owns the
global decoding decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pymatching
import torch
from clifft import noncomp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.loss_aware_teacher import local_erasure_teacher_targets
from evaluation.loss_aware_predecoder import residual_detectors_and_frame
from model.predecoder import PreDecoderModelMemory_v2
from qec.surface_code.deltakit_loss import memory_measurements_to_v2_input
from qec.surface_code.memory_circuit import MemoryCircuit


def _transition_matrix(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    matrix = [[0.0] * 5 for _ in range(5)]
    for (destination, source), probability in entries.items():
        matrix[destination][source] = probability
    return matrix


def _loss_classifier() -> noncomp.Classifier:
    """Keep binary readout, using Clifft's third outcome as a loss herald."""
    level = noncomp.Level
    matrix = [[0.0] * 5 for _ in range(3)]
    matrix[0][level.G] = matrix[1][level.E] = 1.0
    matrix[0][level.LEAK_G] = matrix[1][level.LEAK_E] = 1.0
    matrix[2][level.LOST] = 1.0
    return noncomp.Classifier(matrix)


def make_clifft_model(leakage_probability: float, loss_probability: float) -> noncomp.Model:
    """A state-independent gate-triggered five-level model for Clifft.

    Equal computational-source transition probabilities select Clifft's exact
    classical trajectory treatment for this simplified CX loss/leakage model.
    """
    if leakage_probability + loss_probability > 1:
        raise ValueError("leakage-p plus loss-p must be at most one.")
    level = noncomp.Level
    transitions = _transition_matrix(
        {
            (level.LEAK_G, level.G): leakage_probability,
            (level.LEAK_G, level.E): leakage_probability,
            (level.LOST, level.G): loss_probability,
            (level.LOST, level.E): loss_probability,
        }
    )
    return noncomp.Model(
        initial_state=[1, 0, 0, 0, 0],
        transitions={"CX": transitions},
        classifier=_loss_classifier(),
    )


def make_memory_circuit(distance: int, rounds: int, pauli_probability: float) -> MemoryCircuit:
    """Use the same MemoryCircuit layout expected by the Ising predecoder."""
    return MemoryCircuit(
        distance=distance,
        idle_error=pauli_probability,
        sqgate_error=pauli_probability,
        tqgate_error=pauli_probability,
        spam_error=(2.0 / 3.0) * pauli_probability,
        n_rounds=rounds,
        basis="X",
        code_rotation="XV",
        add_boundary_detectors=True,
    )


@dataclass(frozen=True)
class Batch:
    train_x: torch.Tensor
    detectors: np.ndarray
    observables: np.ndarray


def sample_batch(
    circuit: MemoryCircuit,
    model: noncomp.Model,
    *,
    shots: int,
    seed: int,
) -> Batch:
    """Sample Clifft records and append only the data-loss herald channel."""
    result = noncomp.sample(str(circuit.stim_circuit), model, shots=shots, seed=seed)
    measurements = np.asarray(result.measurements, dtype=np.uint8)
    heralds = np.asarray(result.heralds, dtype=np.uint8)
    data_count = circuit.distance * circuit.distance
    # MemoryCircuit ends with a single data-qubit measurement block.  A lost
    # atom is still lost at that final readout, so this is the available
    # post-block loss herald. Broadcast it across rounds for the CNN's local
    # space-time input shape while retaining the standard first four channels.
    final_data_herald = heralds[:, -data_count:].reshape(shots, circuit.distance, circuit.distance)
    loss_channel = np.broadcast_to(
        final_data_herald[:, None],
        (shots, circuit.n_rounds, circuit.distance, circuit.distance),
    ).copy()
    train_x = memory_measurements_to_v2_input(
        measurements,
        loss_channel,
        distance=circuit.distance,
        n_rounds=circuit.n_rounds,
        basis="X",
        code_rotation="XV",
    )
    return Batch(
        train_x=train_x,
        detectors=np.asarray(result.detectors, dtype=np.uint8),
        observables=np.asarray(result.observables, dtype=np.uint8),
    )


def make_config(distance: int, rounds: int) -> SimpleNamespace:
    return SimpleNamespace(
        distance=distance,
        n_rounds=rounds,
        model=SimpleNamespace(
            input_channels=5,
            out_channels=4,
            dropout_p=0.0,
            activation="gelu",
            num_filters=[128, 128, 128, 4],
            kernel_size=[3, 3, 3, 3],
        ),
    )


def train_local_predecoder(
    model: nn.Module,
    train_x: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> None:
    targets = local_erasure_teacher_targets(train_x, code_rotation="XV")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_x, targets), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
        print(f"epoch {epoch + 1}/{epochs}: local-teacher BCE={total_loss / len(train_x):.6f}")


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    return float(np.any(predictions != observables, axis=1).mean())


def make_matching(circuit: MemoryCircuit, loss_boundary_weight: float) -> pymatching.Matching:
    """Build the ordinary Pauli DEM graph plus a finite-cost loss fallback.

    Clifft loss can produce an otherwise impossible odd detector component.
    The extra detector-to-boundary edges keep PyMatching defined for those
    syndromes; existing Pauli edges remain unchanged and are preferred when
    their lower total weight explains the event.
    """
    matching = pymatching.Matching.from_detector_error_model(
        circuit.stim_circuit.detector_error_model(decompose_errors=True)
    )
    for detector in range(matching.num_detectors):
        matching.add_boundary_edge(
            detector,
            weight=loss_boundary_weight,
            merge_strategy="independent",
        )
    return matching


def evaluate(
    model: nn.Module,
    batch: Batch,
    matching: pymatching.Matching,
    *,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compare standard PyMatching with the same graph after local reduction."""
    baseline_prediction = matching.decode_batch(batch.detectors)
    model.eval()
    with torch.no_grad():
        local_prediction = (torch.sigmoid(model(batch.train_x.to(device))) >= 0.5).to(torch.uint8)
    residual, local_frame = residual_detectors_and_frame(
        local_prediction,
        batch.detectors,
        np.arange(batch.detectors.shape[1], dtype=np.intp),
        basis="X",
        code_rotation="XV",
        output_semantics="residual",
    )
    predecoded_prediction = matching.decode_batch(residual)
    predecoded_prediction[:, 0] ^= local_frame
    baseline_ler = logical_error_rate(baseline_prediction, batch.observables)
    predecoded_ler = logical_error_rate(predecoded_prediction, batch.observables)
    average_local_corrections = float(local_prediction[:, :2].float().mean().cpu())
    return baseline_ler, predecoded_ler, average_local_corrections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--train-shots", type=int, default=4_096)
    parser.add_argument("--test-shots", type=int, default=1_024)
    parser.add_argument("--pauli-p", type=float, default=0.002)
    parser.add_argument("--leakage-p", type=float, default=0.001)
    parser.add_argument("--loss-p", type=float, default=0.004)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--loss-boundary-weight",
        type=float,
        default=2.0,
        help="PyMatching cost for the fallback edge used only by Clifft loss syndromes.",
    )
    args = parser.parse_args()
    if args.distance < 3 or args.distance % 2 == 0:
        raise ValueError("distance must be an odd integer at least three.")
    if args.rounds < 2:
        raise ValueError("rounds must be at least two.")

    circuit = make_memory_circuit(args.distance, args.rounds, args.pauli_p)
    clifft_model = make_clifft_model(args.leakage_p, args.loss_p)
    train_batch = sample_batch(circuit, clifft_model, shots=args.train_shots, seed=args.seed)
    test_batch = sample_batch(circuit, clifft_model, shots=args.test_shots, seed=args.seed + 1)
    matching = make_matching(circuit, args.loss_boundary_weight)
    device = torch.device(args.device)
    model = PreDecoderModelMemory_v2(make_config(args.distance, args.rounds)).to(device)
    train_local_predecoder(
        model,
        train_batch.train_x,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    baseline_ler, predecoded_ler, average_local_corrections = evaluate(
        model, test_batch, matching, device=device
    )
    print(f"features: {tuple(train_batch.train_x.shape)} = [X, Z, X-geometry, Z-geometry, loss-herald]")
    print(f"Clifft sample: detectors={test_batch.detectors.shape[1]}, observables={test_batch.observables.shape[1]}")
    print(f"PyMatching baseline logical error rate: {baseline_ler:.3%}")
    print(f"loss-aware local predecoder + PyMatching logical error rate: {predecoded_ler:.3%}")
    print(f"mean predicted local correction density: {average_local_corrections:.6f}")


if __name__ == "__main__":
    main()
