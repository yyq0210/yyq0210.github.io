"""CPU schedule model of matmul_v3.cu, not a CUDA correctness test.

Run with Python 3; writes verification.json and schedule.csv beside this file.
This models ordered submissions/completion waits, NOT hardware memory semantics.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_schedule(stages, tiles):
    if stages < 1 or tiles < max(1, stages - 1):
        raise ValueError("Outside the original loop's supported tile count")
    slots = [None] * stages
    phase_array = [0] * stages
    tma_phase = mma_phase = 0
    loaded, computed, completed = set(), set(), set()
    events = []
    mma_k_ranges = []
    initializations = 0

    def load(i, section):
        assert 0 <= i < tiles and i not in loaded
        stage = i % stages
        old = slots[stage]
        assert old is None or old in completed, (i, old, "premature overwrite")
        slots[stage] = i
        loaded.add(i)
        events.append(dict(section=section, event="load", tile=i, stage=stage,
                           tma_wait_parity="", mma_wait_parity="", replaces=old))

    def compute_and_wait(i, section):
        nonlocal tma_phase, mma_phase, initializations
        stage = i % stages
        assert slots[stage] == i and i in loaded and i not in computed
        assert tma_phase == phase_array[stage] == (i // stages) % 2
        current_tma_phase = tma_phase
        phase_array[stage] ^= 1
        if stage == stages - 1:
            tma_phase ^= 1
        for j in range(8):
            k1, k2 = divmod(j, 4)
            k_start = i * 128 + k1 * 64 + k2 * 16
            mma_k_ranges.append((k_start, k_start + 16))
            accumulate = bool(i) if j == 0 else True
            initializations += int(not accumulate)
            # Descriptor start for the first row; subsequent rows are swizzled.
            a_base = stage * 98304
            b_base = a_base + 32768
            a_desc = a_base + k1 * 128 * 128 + k2 * 32
            b_desc = b_base + k1 * 256 * 128 + k2 * 32
            assert a_base <= a_desc < a_base + 32768
            assert b_base <= b_desc < b_base + 65536
        computed.add(i)
        events.append(dict(section=section, event="compute_issue", tile=i, stage=stage,
                           tma_wait_parity=current_tma_phase, mma_wait_parity="", replaces=""))
        assert mma_phase == i % 2
        completed.add(i)
        events.append(dict(section=section, event="mma_wait_complete", tile=i, stage=stage,
                           tma_wait_parity="", mma_wait_parity=mma_phase, replaces=""))
        mma_phase ^= 1

    for i in range(stages - 1):
        load(i, "prefetch")
    for i in range(tiles - stages + 1):
        load(i + stages - 1, "main")
        compute_and_wait(i, "main")
    for i in range(tiles - stages + 1, tiles):
        compute_and_wait(i, "drain")
    assert loaded == computed == completed == set(range(tiles))
    assert mma_k_ranges == [(k, k + 16) for k in range(0, tiles * 128, 16)]
    assert initializations == 1
    return events


def main():
    # Check the per-stage swizzled layout is a bijection over BF16 addresses.
    for height in (128, 256):
        addresses = set()
        for r in range(height):
            for q in range(128):
                offset = ((q // 64) * height * 128 + r * 128
                          + (((q % 64) // 8) ^ (r % 8)) * 16 + (q % 8) * 2)
                addresses.add(offset)
        assert addresses == set(range(0, height * 128 * 2, 2))
    cases = 0
    for stages in range(1, 9):
        for tiles in range(max(1, stages - 1), 65):
            check_schedule(stages, tiles)
            cases += 1
    rejected = 0
    for stages in range(1, 9):
        for tiles in range(max(1, stages - 1)):
            try:
                check_schedule(stages, tiles)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Invalid loop boundary was accepted")
    events = check_schedule(2, 32)
    with (ROOT / "schedule.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    ratios = {
        "b300_v3_over_v2b": 1039 / 764,
        "b300_throughput_gain_percent": (1039 / 764 - 1) * 100,
        "b300_v3_percent_of_cublas": 1039 / 1673 * 100,
        "b200_v3_over_v2b": 939.61 / 695.43,
        "b300_vs_b200_v3_gain_percent": (1039 / 939.61 - 1) * 100,
        "inferred_equal_work_time_reduction_percent": (1 - 764 / 1039) * 100,
    }
    assert round(ratios["b300_throughput_gain_percent"], 1) == 36.0
    assert round(ratios["b300_v3_percent_of_cublas"], 1) == 62.1
    assert 98304 + 16384 + 64 == 114752
    assert 131072 + 32768 + 64 == 163904
    report = {
        "status": "passed",
        "scope": "CPU ordered schedule, address arithmetic, phase parity and benchmark arithmetic only",
        "not_verified": ["CUDA compilation", "GPU execution", "hardware memory ordering", "performance reproduction"],
        "valid_schedule_cases": cases,
        "rejected_invalid_boundaries": rejected,
        "default": {"stages": 2, "tiles": 32, "loads": 32, "computes": 32,
                    "mma_instructions_per_block": 256, "dynamic_smem_bytes": 196608,
                    "grid_blocks_for_4096_cubed": 512},
        "ratios": ratios,
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
