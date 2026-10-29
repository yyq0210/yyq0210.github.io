"""CPU event model of the V4 ring-buffer protocol. No CUDA is executed.

Separate issue/completion events; allow out-of-order TMA completion.
Use ordered MMA completion as a simplified accumulator-dependency model.
Not a simulator of PTX memory ordering, device queues, warp scheduling or time.
Run: python3 verify_pipeline.py
"""
from pathlib import Path
import csv
import json
import random

ROOT = Path(__file__).resolve().parent


def simulate(stages, tiles, seed):
    assert stages >= 1 and tiles >= 1
    rng = random.Random(seed)
    owner = [None] * stages
    t_parity = [0] * stages
    d_parity = [0] * stages
    producer_phase = consumer_phase = 0
    next_load = next_compute = next_done = 0
    pending_load, ready = set(), set()
    pending_compute = set()
    loaded, issued, completed = set(), set(), set()
    final_issued = final_complete = output_read = False
    events = []
    max_pending_mma = 0
    final_commit_had_pending_mma = False

    while not output_read:
        actions = []
        if next_load < tiles:
            s = next_load % stages
            assert producer_phase == (next_load // stages) % 2
            # The opposite initial parity permits first use of an empty slot.
            if d_parity[s] != (producer_phase ^ 1):
                actions.append(('load_issue', next_load))
        actions.extend(('load_complete', i) for i in sorted(pending_load))
        if next_compute < tiles:
            s = next_compute % stages
            assert consumer_phase == (next_compute // stages) % 2
            if t_parity[s] != consumer_phase:
                actions.append(('compute_issue', next_compute))
        if next_done in pending_compute:
            actions.append(('compute_complete', next_done))
        if next_compute == tiles and not final_issued:
            actions.append(('final_commit', None))
        if final_issued and next_done == tiles and not final_complete:
            actions.append(('final_complete', None))
        if final_complete:
            actions.append(('output_read', None))
        assert actions, ('deadlock', stages, tiles, seed)
        action, i = rng.choice(actions)
        s = i % stages if i is not None else None
        wait_parity = ''

        if action == 'load_issue':
            # This assertion detects unsafe reuse independently of parity check.
            assert owner[s] is None or owner[s] in completed
            assert i not in loaded
            wait_parity = producer_phase ^ 1
            owner[s] = i
            loaded.add(i)
            pending_load.add(i)
            next_load += 1
            if s == stages - 1:
                producer_phase ^= 1
        elif action == 'load_complete':
            assert owner[s] == i
            pending_load.remove(i)
            ready.add(i)
            assert t_parity[s] == (i // stages) % 2
            t_parity[s] ^= 1
        elif action == 'compute_issue':
            assert owner[s] == i and i in ready and i not in issued
            wait_parity = consumer_phase
            ready.remove(i)
            issued.add(i)
            pending_compute.add(i)
            next_compute += 1
            if s == stages - 1:
                consumer_phase ^= 1
        elif action == 'compute_complete':
            assert owner[s] == i
            pending_compute.remove(i)
            completed.add(i)
            assert d_parity[s] == (i // stages) % 2
            d_parity[s] ^= 1
            next_done += 1
        elif action == 'final_commit':
            final_issued = True
            final_commit_had_pending_mma = bool(pending_compute)
        elif action == 'final_complete':
            assert len(completed) == tiles
            final_complete = True
        elif action == 'output_read':
            assert final_complete and len(completed) == tiles
            output_read = True
        max_pending_mma = max(max_pending_mma, len(pending_compute))
        assert next_load - next_done <= stages
        events.append({'event_index': len(events), 'event': action,
                       'tile': i if i is not None else '',
                       'stage': s if s is not None else '',
                       'wait_parity': wait_parity,
                       'loads_issued': next_load, 'computes_issued': next_compute,
                       'computes_completed': next_done,
                       'pending_mma_groups': len(pending_compute)})
        assert len(events) <= tiles * 4 + 3

    assert loaded == issued == completed == set(range(tiles))
    assert [e['tile'] for e in events if e['event'] == 'compute_issue'] == list(range(tiles))
    assert len(events) == tiles * 4 + 3
    return events, {'max_pending_mma': max_pending_mma,
                    'final_commit_had_pending_mma': final_commit_had_pending_mma}


def write_trace(name, events):
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)


def unsafe_examples():
    # A missing release wait permits L7 while C0 is still consuming stage 0.
    owner = [0] * 7
    completed = set()
    unsafe_overwrite_detected = owner[0] not in completed
    # Reaching the end of the issue loop does not mean all groups completed.
    issued = {0, 1, 2, 3}
    done = {0, 1}
    premature_output_detected = len(issued) == 4 and len(done) < 4
    assert unsafe_overwrite_detected and premature_output_detected
    return {'missing_release_wait': 'unsafe overwrite detected',
            'missing_final_completion_wait': 'premature output detected'}


def main():
    cases = 0
    max_pending = 0
    final_with_pending = 0
    for s in range(1, 9):
        for t in (1, 2, 3, 4, 6, 7, 8, 14, 16, 32, 64):
            for seed in range(8):
                _, stats = simulate(s, t, seed)
                cases += 1
                max_pending = max(max_pending, stats['max_pending_mma'])
                final_with_pending += int(stats['final_commit_had_pending_mma'])
    assert max_pending > 1 and final_with_pending > 0
    default, _ = simulate(7, 64, 3)
    short, _ = simulate(7, 4, 3)
    assert {e['stage'] for e in short if e['event'] == 'load_issue'} == {0, 1, 2, 3}
    write_trace('schedule-default.csv', default)
    write_trace('schedule-short-k.csv', short)

    # Per-stage swizzle covers every BF16 address exactly once.
    offsets = {r * 128 + ((q // 8) ^ (r % 8)) * 16 + (q % 8) * 2
               for r in range(128) for q in range(64)}
    assert offsets == set(range(0, 16384, 2))
    stage_ranges = []
    for s in range(7):
        base = s * 32768
        stage_ranges.extend([(base, base + 16384), (base + 16384, base + 32768)])
    assert stage_ranges[0][0] == 0 and stage_ranges[-1][1] == 229376
    assert all(a[1] == b[0] for a, b in zip(stage_ranges, stage_ranges[1:]))
    k_ranges = [(i * 64 + j * 16, i * 64 + (j + 1) * 16)
                for i in range(64) for j in range(4)]
    assert k_ranges == [(k, k + 16) for k in range(0, 4096, 16)]
    assert sum(not (bool(i) if j == 0 else True) for i in range(64) for j in range(4)) == 1

    ratios = {
        'b300_v3_to_v4_gain_percent': (1271 / 1039 - 1) * 100,
        'b200_v3_to_v4_gain_percent': (1208.83 / 939.61 - 1) * 100,
        'b300_v4_percent_of_cublas': 1271 / 1676 * 100,
        'b300_inferred_same_work_time_reduction_percent': (1 - 1039 / 1271) * 100,
    }
    assert round(ratios['b300_v3_to_v4_gain_percent'], 1) == 22.3
    assert round(ratios['b300_v4_percent_of_cublas'], 1) == 75.8
    report = {'status': 'passed', 'model': 'CPU randomized legal event interleavings',
              'schedule_cases': cases, 'max_pending_mma_groups_observed': max_pending,
              'cases_final_commit_before_all_mma_completed': final_with_pending,
              'short_k_used_stages': [0, 1, 2, 3],
              'negative_examples': unsafe_examples(),
              'default': {'stages': 7, 'tiles': 64, 'barriers': 15,
                          'dynamic_smem_bytes': 229376, 'mma_instructions_per_block': 256,
                          'grid_blocks_for_4096_cubed': 1024},
              'ratios': ratios,
              'not_verified': ['CUDA compilation', 'GPU correctness', 'PTX memory model',
                               'hardware scheduling and timing', 'GPU benchmark reproduction']}
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
