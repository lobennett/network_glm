import numpy as np
import pandas as pd
from typing import Optional

def stop_fail_violation(events: pd.DataFrame,
        trial_type: str = "trial_type",
        rt: str = "rt",
        go_acc: str = "go_acc",
        output: str = "stop_failure_violation",
        go: str = "go",
        stop_failure: str = "stop_failure",
        min_go_rt: float = 0.2,
        is_test_trial: Optional[str] = None) -> pd.DataFrame:
    """
    Add a per-run column with the violation amplitude for stop_failure trials
    For each stop_failure trial:
      amplitude = stop_failure_rt of trial N - valid_go_rt of trial N-1
    """
    df = events.copy()
    df[output] = np.nan

    def check_if_test_trial(idx: int) -> bool:  #Returns true if the row is considered a test trial
        if is_test_trial and is_test_trial in df.columns:
            return bool(df.at[idx, is_test_trial])
        return True  #assumes row is a test trial if not specified otherwise

    def prev_is_valid_go(prev_row: pd.Series) -> bool:
        if prev_row.get(trial_type) != go:  #checks hat it's a go trial
            return False

        if go_acc in prev_row.index: #check go_acc == 1
            valid_go_acc = prev_row.get(go_acc)
            if pd.isna(valid_go_acc):
                return False
            try:
                if float(valid_go_acc) != 1.0:
                    return False
            except Exception:
                return False
        else:
            return False

        prev_rt = prev_row.get(rt)  #RT invalid if -1 or NaN
        if pd.isna(prev_rt):
            return False
        if prev_rt == -1:  #omission
            return False
        if prev_rt < min_go_rt:  #too fast
            return False

        return True

    amplitudes = []
    amp_indices = []
    prev_test_idx = None

    for idx in range(len(df)):
        if not check_if_test_trial(idx):  #skips non_test trials
            continue

        row = df.loc[idx]
        cur_trial_type = row.get(trial_type)

        if cur_trial_type == stop_failure:
            if prev_test_idx is None:
                prev_test_idx = idx
                continue

            prev_row = df.loc[prev_test_idx]

            cur_rt = row.get(rt)
            if pd.isna(cur_rt):
                prev_test_idx = idx
                continue

            if prev_is_valid_go(prev_row):
                amp = float(cur_rt) - float(prev_row.get(rt))
                amplitudes.append(amp)
                amp_indices.append(idx)

        prev_test_idx = idx

    if amplitudes:
        mean_amp = float(np.mean(amplitudes))
        centered = [each - mean_amp for each in amplitudes]
        for i, v in zip(amp_indices, centered):
            df.at[i, output] = v

    return df
