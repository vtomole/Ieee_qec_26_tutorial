#!/usr/bin/env python3
"""Train a four-channel local neural predecoder on Stim Pauli-noise samples.

This is the loss-free half of the tutorial paired with
``clifft_ising_loss_aware_predecoder.py``:

    Stim FlipSimulator -> [X syndrome, Z syndrome, X geometry, Z geometry]
        -> four-head local CNN -> residual detector record -> PyMatching

Stim's FlipSimulator exposes the hidden Pauli frame after each syndrome round.
Those frames provide supervised correction/residual targets; they are never
available to the model at inference time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pymatching
import stim
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.loss_aware_circuit_teacher import circuit_frame_teacher_targets
from evaluation.loss_aware_predecoder import residual_detectors_and_frame
from model.predecoder import PreDecoderModelMemory_v1
from qec.surface_code.deltakit_loss import memory_measurements_to_v2_input
from qec.surface_code.memory_circuit import MemoryCircuit


@dataclass(frozen=True)
class Batch:
    model_x: torch.Tensor
    formatter_x: torch.Tensor
    teacher_y: torch.Tensor
    detectors: np.ndarray
    observables: np.ndarray


def make_memory_circuit(distance: int, rounds: int, pauli_probability: float) -> MemoryCircuit:
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


def sample_stim_batch(circuit: MemoryCircuit, *, shots: int, seed: int) -> Batch:
    """Sample Pauli noise with Stim and retain hidden data frames as labels."""
    simulator = stim.FlipSimulator(
        batch_size=shots,
        num_qubits=circuit.stim_circuit.num_qubits,
        seed=seed,
        disable_stabilizer_randomization=True,
    )
    x_frames: list[np.ndarray] = []
    z_frames: list[np.ndarray] = []
    data_qubits = np.asarray(circuit.code.data_qubits, dtype=np.intp)
    for instruction in circuit.stim_circuit.flattened():
        simulator.do(instruction)
        # In MemoryCircuit, this reset-measurement completes one syndrome round.
        if instruction.name == "MR":
            x, z, _, _, _ = simulator.to_numpy(
                transpose=True, output_xs=True, output_zs=True
            )
            x_frames.append(x[:, data_qubits])
            z_frames.append(z[:, data_qubits])
    if len(x_frames) != circuit.n_rounds:
        raise RuntimeError(f"Expected {circuit.n_rounds} syndrome-round frame snapshots, got {len(x_frames)}.")

    _, _, measurements, detectors, observables = simulator.to_numpy(
        transpose=True,
        output_measure_flips=True,
        output_detector_flips=True,
        output_observable_flips=True,
    )
    zero_loss_herald = np.zeros(
        (shots, circuit.n_rounds, circuit.distance, circuit.distance), dtype=np.uint8
    )
    # The common formatter is five-channel. Channel 5 is identically zero in
    # this loss-free lesson and is omitted before passing data to the v1 model.
    formatter_x = memory_measurements_to_v2_input(
        measurements.astype(np.uint8, copy=False),
        zero_loss_herald,
        distance=circuit.distance,
        n_rounds=circuit.n_rounds,
        basis="X",
        code_rotation="XV",
    )
    data_x_frames = np.stack(x_frames, axis=1).reshape(
        shots, circuit.n_rounds, circuit.distance, circuit.distance
    )
    data_z_frames = np.stack(z_frames, axis=1).reshape(
        shots, circuit.n_rounds, circuit.distance, circuit.distance
    )
    teacher_y = circuit_frame_teacher_targets(
        formatter_x,
        torch.from_numpy(data_x_frames),
        torch.from_numpy(data_z_frames),
        code_rotation="XV",
    )
    return Batch(
        model_x=formatter_x[:, :4].contiguous(),
        formatter_x=formatter_x,
        teacher_y=teacher_y,
        detectors=detectors.astype(np.uint8, copy=False),
        observables=observables.astype(np.uint8, copy=False),
    )


def make_config(distance: int, rounds: int) -> SimpleNamespace:
    return SimpleNamespace(
        distance=distance,
        n_rounds=rounds,
        model=SimpleNamespace(
            input_channels=4,
            out_channels=4,
            dropout_p=0.0,
            activation="gelu",
            num_filters=[128, 128, 128, 4],
            kernel_size=[3, 3, 3, 3],
        ),
    )


def train(model: nn.Module, batch: Batch, *, device: torch.device, epochs: int, batch_size: int) -> None:
    loader = DataLoader(TensorDataset(batch.model_x, batch.teacher_y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
        print(f"epoch {epoch + 1}/{epochs}: circuit-frame BCE={total_loss / len(batch.model_x):.6f}")


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    return float(np.any(predictions != observables, axis=1).mean())


def evaluate(model: nn.Module, batch: Batch, matching: pymatching.Matching, *, device: torch.device) -> tuple[float, float]:
    baseline = matching.decode_batch(batch.detectors)
    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(batch.model_x.to(device))) >= 0.5).to(torch.uint8)
    residual, local_frame = residual_detectors_and_frame(
        prediction,
        batch.detectors,
        np.arange(batch.detectors.shape[1], dtype=np.intp),
        basis="X",
        code_rotation="XV",
        train_x=batch.formatter_x,
        output_semantics="timelike",
    )
    predecoded = matching.decode_batch(residual)
    predecoded[:, 0] ^= local_frame
    return logical_error_rate(baseline, batch.observables), logical_error_rate(predecoded, batch.observables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--train-shots", type=int, default=4_096)
    parser.add_argument("--test-shots", type=int, default=1_024)
    parser.add_argument("--pauli-p", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.distance < 3 or args.distance % 2 == 0:
        raise ValueError("distance must be an odd integer at least three.")
    if args.rounds < 2:
        raise ValueError("rounds must be at least two.")

    circuit = make_memory_circuit(args.distance, args.rounds, args.pauli_p)
    train_batch = sample_stim_batch(circuit, shots=args.train_shots, seed=args.seed)
    test_batch = sample_stim_batch(circuit, shots=args.test_shots, seed=args.seed + 1)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.stim_circuit.detector_error_model(decompose_errors=True)
    )
    model = PreDecoderModelMemory_v1(make_config(args.distance, args.rounds)).to(args.device)
    train(model, train_batch, device=torch.device(args.device), epochs=args.epochs, batch_size=args.batch_size)
    baseline_ler, predecoded_ler = evaluate(model, test_batch, matching, device=torch.device(args.device))
    print(f"features: {tuple(train_batch.model_x.shape)} = [X, Z, X-geometry, Z-geometry]")
    print("simulator: Stim FlipSimulator (circuit-level Pauli noise; no loss herald)")
    print(f"PyMatching baseline logical error rate: {baseline_ler:.3%}")
    print(f"Stim-trained local predecoder + PyMatching logical error rate: {predecoded_ler:.3%}")


if __name__ == "__main__":
    main()
