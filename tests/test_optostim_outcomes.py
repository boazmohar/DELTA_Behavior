from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.processing.optostim import build_optostim_trials


def test_build_optostim_trials_normalizes_and_inferrs_outcomes() -> None:
    action_log = pd.DataFrame(
        [
            # Trial 1: Go, lick present -> inferred Hit (outcome missing)
            {"time_ms": 0, "action": "TrialStart", "trialNumber": 1, "trialType": "Go"},
            {"time_ms": 100, "action": "StimulusDelivered", "trialNumber": 1},
            {"time_ms": 150, "action": "LickDetected", "trialNumber": 1},
            {"time_ms": 500, "action": "TrialEnd", "trialNumber": 1, "outcome": None, "rewarded": True},
            # Trial 2: Go, no lick -> inferred Miss
            {"time_ms": 1000, "action": "TrialStart", "trialNumber": 2, "trialType": "Go"},
            {"time_ms": 1100, "action": "StimulusDelivered", "trialNumber": 2},
            {"time_ms": 1500, "action": "TrialEnd", "trialNumber": 2, "outcome": "", "rewarded": False},
            # Trial 3: NoGo, immediate false alarm outcome label -> mapped to FA
            {"time_ms": 2000, "action": "TrialStart", "trialNumber": 3, "trialType": "NoGo"},
            {"time_ms": 2100, "action": "StimulusDelivered", "trialNumber": 3},
            {"time_ms": 2200, "action": "ImmediateFalseAlarm", "trialNumber": 3},
            {"time_ms": 2500, "action": "TrialEnd", "trialNumber": 3, "outcome": "ImmediateFalseAlarm", "rewarded": False},
            # Trial 4: NoGo, no lick -> inferred CR
            {"time_ms": 3000, "action": "TrialStart", "trialNumber": 4, "trialType": "NoGo"},
            {"time_ms": 3100, "action": "StimulusDelivered", "trialNumber": 4},
            {"time_ms": 3500, "action": "TrialEnd", "trialNumber": 4, "outcome": None, "rewarded": False},
        ]
    )

    trials = build_optostim_trials(action_log)
    assert len(trials) == 4
    assert set(trials["outcome"]) <= {"Hit", "Miss", "FA", "CR"}

    outcomes_by_trial = trials.set_index("trial_number")["outcome"].to_dict()
    assert outcomes_by_trial[1] == "Hit"
    assert outcomes_by_trial[2] == "Miss"
    assert outcomes_by_trial[3] == "FA"
    assert outcomes_by_trial[4] == "CR"

    sources_by_trial = trials.set_index("trial_number")["outcome_source"].to_dict()
    assert sources_by_trial[1] == "inferred"
    assert sources_by_trial[2] == "inferred"
    assert sources_by_trial[3] == "raw"
    assert sources_by_trial[4] == "inferred"

