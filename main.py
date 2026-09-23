import os
import multiprocessing as mp
import sys

from SSVEP_stimulator import SSVEPStimulator
from util import SharedState

sys.path.append(".")


def main():
    """Run the continuous same-frequency phase-coded SSVEP task."""

    # ---------------- User-adjustable experiment parameters ----------------
    online = False
    task_m = 2
    paradigm_mode = "active"  # "active" or "passive"; passive requires task_m=2
    block_count = 10
    cue_duration_min = 3.0
    cue_duration_max = 5.0
    stim_square_scale_by_m = {
        2: 0.22,
        4: 0.22,
        9: 0.22,
    }
    stim_gap_scale_by_m = {
        2: 0.45,
        4: 0.45,
        9: 0.45,
    }

    # ---------------- Decoder / acquisition parameters ----------------
    stim_freq = 10.0
    fs = 2000
    window_size = 104
    min_samples_per_class = 20
    training_skip_after_cue_ms = 500.0
    update_stride = 20
    fusion_window_edges = 1
    min_decode_interval_ms = 100.0
    debug_decoder = False

    share = SharedState(
        window_size=window_size,
        frequency=stim_freq,
        fs=fs,
        task_m=task_m,
        paradigm_mode=paradigm_mode,
        online=online,
        block_count=block_count,
        cue_duration_min=cue_duration_min,
        cue_duration_max=cue_duration_max,
        min_samples_per_class=min_samples_per_class,
        training_skip_after_cue_ms=training_skip_after_cue_ms,
        update_stride=update_stride,
        fusion_window_edges=fusion_window_edges,
        min_decode_interval_ms=min_decode_interval_ms,
        debug_decoder=debug_decoder,
    )

    stim = SSVEPStimulator(
        shared_state=share,
        required_refresh_rate=600,
        square_scale_by_m=stim_square_scale_by_m,
        gap_scale_by_m=stim_gap_scale_by_m,
    )
    stim.run()
    print("Main: Application exited.", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")
    main()
