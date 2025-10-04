import sys
from pathlib import Path

import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

pytestmark = pytest.mark.filterwarnings(
    "ignore:np\\.find_common_type is deprecated:DeprecationWarning"
)

from behavioral_analysis.processing import (  # noqa: E402
    add_corridor_info_to_events,
    compute_corridor_artifacts,
)


def _make_mock_session():
    # Path position covering two corridors; first loop ends at time 20 while
    # the last cue result happens later (time 22) to exercise corridor end logic.
    path_df = pd.DataFrame({
        'time': [0, 5, 10, 15, 20, 24, 28, 32, 36, 40],
        'position': [0, 10000, 50000, 10000, 500, 0, 28000, 48000, 20000, 100],
    })

    cue_states = pd.DataFrame({
        'time': [1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30],
        'id': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
        'position': [500, 5000, 10000, 15000, 20000, 26000, 32000,
                     600, 5600, 11200, 16800, 22400, 28000, 33600],
        'isRewarding': [True, False, True, True, True, False, True,
                        True, True, True, True, False, True, True],
    })

    cue_results = pd.DataFrame({
        'time': [11, 12, 13, 14, 15, 16, 26, 34, 35, 36, 37, 38, 39, 40],
        'id': cue_states['id'],
        'position': cue_states['position'],
        'isRewarding': cue_states['isRewarding'],
        'hasGivenReward': [True, False, True, True, True, False, True,
                           True, True, True, True, False, True, True],
        'numLicksInReward': [1] * 14,
        'numLicksInPre': [0] * 14,
    })

    return path_df, cue_states, cue_results


def test_corridor_end_time_extends_past_latest_cue():
    path_df, cue_states, cue_results = _make_mock_session()

    artifacts = compute_corridor_artifacts(
        cue_state_df=cue_states,
        position_df=path_df,
        cue_result_df=cue_results,
        verbose=False,
    )

    corridor_info = artifacts.corridor_info

    # Corridor 0 should report the late cue result at time 26 in the summary
    first_corridor = corridor_info.loc[corridor_info['corridor_id'] == 0].iloc[0]
    assert first_corridor['last_cue_time'] == 26
    assert first_corridor['trigger'] == 'first_cue'
    assert first_corridor['num_cue_results'] == 7

    second_corridor = corridor_info.loc[corridor_info['corridor_id'] == 1].iloc[0]
    assert second_corridor['trigger'] == 'cue_reset'
    assert second_corridor['num_cue_results'] == 7
    # Ensure corridor windows stay ordered
    assert first_corridor['end_time'] <= second_corridor['start_time']

    # Matched cue counts should be available for debugging/QA
    assert first_corridor['num_matched_cues'] == 7
    assert second_corridor['num_matched_cues'] == 7


def test_event_assignment_respects_correct_corridor():
    path_df, cue_states, cue_results = _make_mock_session()

    artifacts = compute_corridor_artifacts(
        cue_state_df=cue_states,
        position_df=path_df,
        cue_result_df=cue_results,
        verbose=False,
    )
    corridor_info = artifacts.corridor_info

    frames = {
        'Cue State': cue_states.copy(),
        'Cue Result': cue_results.copy(),
        'Path Position': path_df.copy(),
    }

    updated = add_corridor_info_to_events(
        frames,
        corridor_info,
        corridor_length_cm=500.0,
        verbose=False,
        position_df=path_df,
    )

    cue_results_labeled = updated['Cue Result']

    # The last cue result in the first corridor should still be labelled 0
    assert cue_results_labeled.loc[6, 'corridor_id'] == 0

    expected_corridors = [0] * 7 + [1] * 7
    assert list(cue_results_labeled['corridor_id']) == expected_corridors
