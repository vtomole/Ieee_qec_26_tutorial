import numpy as np
import pymatching
import deltakit_stim
import stim


def count_logical_errors(
    detector_error_model: deltakit_stim.DetectorErrorModel,
    num_shots: int,
) -> int:
    """Return the number of logical-decoder failures in samples from a DEM.

    The detector error model already includes the effects of the circuit noise,
    including the leakage-herald detector events produced by
    HERALD_LEAKAGE_EVENT.  Sampling it is therefore useful for evaluating a
    decoder directly against that error model.
    """
    # Sample detector events and the corresponding true logical-observable
    # flips from the detector error model.
    sampler = detector_error_model.compile_sampler()
    samples = sampler.sample(num_shots)
    detection_events, observable_flips = samples[:2]

    # PyMatching expects Stim's Python type, whereas DeltaKit-Stim provides
    # its own compatible DEM type. Convert the serialised DEM at this
    # boundary; sampling above still uses DeltaKit-Stim's leakage-aware DEM.
    pymatching_dem = stim.DetectorErrorModel(str(detector_error_model))
    matcher = pymatching.Matching.from_detector_error_model(pymatching_dem)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

circuit = deltakit_stim.Circuit("""
R 0 1 2
CZ 0 1
CZ 1 2
LEAKAGE(0.1) 1
HERALD_LEAKAGE_EVENT 1
M 0 1 2
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
""")
num_shots = 1000
detector_error_model = circuit.detector_error_model(decompose_errors=True)
print("Detector Error Model:\\n", detector_error_model)

if detector_error_model.num_observables == 0:
    print("No logical observables are declared, so the logical-error count will be zero.")

num_logical_errors = count_logical_errors(detector_error_model, num_shots)
print("there were", num_logical_errors, "wrong predictions (logical errors) out of", num_shots, "shots")
