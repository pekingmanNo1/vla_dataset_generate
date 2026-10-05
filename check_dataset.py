from pathlib import Path
from collections import Counter
import json
import numpy as np


DATASET_ROOT = Path("vla_dataset")


REQUIRED_KEYS = [
    "timestamp",
    "active_cube",
    "cube_position",
    "place_position",
    "joint_position",
    "joint_velocity",
    "action_joint_position",
    "action_joint_velocity",
    "phase",
]


def has_nan_or_inf(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return False

    return not np.all(np.isfinite(array))


def main():
    episode_dirs = sorted(DATASET_ROOT.glob("episode_*"))

    print("=" * 60)
    print("VLA Dataset Validation")
    print("=" * 60)

    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"Total episodes: {len(episode_dirs)}")

    if not episode_dirs:
        print("No episodes found.")
        return

    success_count = 0
    failure_count = 0

    failure_reasons = Counter()

    frame_counts = []

    invalid_episodes = []

    for episode_dir in episode_dirs:
        npz_path = episode_dir / "trajectory.npz"
        metadata_path = episode_dir / "metadata.json"

        episode_errors = []

        # -----------------------------
        # Check files exist
        # -----------------------------
        if not npz_path.exists():
            episode_errors.append("trajectory.npz missing")

        if not metadata_path.exists():
            episode_errors.append("metadata.json missing")

        if episode_errors:
            invalid_episodes.append(
                (episode_dir.name, episode_errors)
            )
            continue

        # -----------------------------
        # Load metadata
        # -----------------------------
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            invalid_episodes.append(
                (episode_dir.name, [f"metadata read error: {e}"])
            )
            continue

        success = bool(metadata.get("success", False))
        failure_reason = metadata.get("failure_reason")
        num_frames_metadata = metadata.get("num_frames")

        if success:
            success_count += 1
        else:
            failure_count += 1
            failure_reasons[
                failure_reason or "Unknown failure"
            ] += 1

        # -----------------------------
        # Load trajectory
        # -----------------------------
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            invalid_episodes.append(
                (episode_dir.name, [f"npz read error: {e}"])
            )
            continue

        # -----------------------------
        # Check required keys
        # -----------------------------
        for key in REQUIRED_KEYS:
            if key not in data.files:
                episode_errors.append(f"missing key: {key}")

        if episode_errors:
            invalid_episodes.append(
                (episode_dir.name, episode_errors)
            )
            continue

        # -----------------------------
        # Frame count
        # -----------------------------
        timestamp = data["timestamp"]
        num_frames = len(timestamp)

        frame_counts.append(num_frames)

        if num_frames == 0:
            episode_errors.append("empty episode")

        if (
            num_frames_metadata is not None
            and num_frames != num_frames_metadata
        ):
            episode_errors.append(
                f"metadata frames={num_frames_metadata}, "
                f"trajectory frames={num_frames}"
            )

        # -----------------------------
        # Check every array has same first dimension
        # -----------------------------
        for key in REQUIRED_KEYS:
            array = data[key]

            if len(array) != num_frames:
                episode_errors.append(
                    f"{key} length={len(array)}, "
                    f"expected={num_frames}"
                )

        # -----------------------------
        # NaN / Inf check
        # -----------------------------
        for key in REQUIRED_KEYS:
            array = data[key]

            if has_nan_or_inf(array):
                episode_errors.append(
                    f"{key} contains NaN or Inf"
                )

        # -----------------------------
        # Shape sanity checks
        # -----------------------------
        if data["cube_position"].ndim != 2:
            episode_errors.append(
                f"cube_position invalid shape "
                f"{data['cube_position'].shape}"
            )

        if data["place_position"].ndim != 2:
            episode_errors.append(
                f"place_position invalid shape "
                f"{data['place_position'].shape}"
            )

        if data["joint_position"].ndim != 2:
            episode_errors.append(
                f"joint_position invalid shape "
                f"{data['joint_position'].shape}"
            )

        if data["action_joint_position"].ndim != 2:
            episode_errors.append(
                f"action_joint_position invalid shape "
                f"{data['action_joint_position'].shape}"
            )

        # -----------------------------
        # Observation/action dimension check
        # -----------------------------
        if (
            data["joint_position"].shape
            != data["action_joint_position"].shape
        ):
            episode_errors.append(
                "joint_position and action_joint_position "
                f"shape mismatch: "
                f"{data['joint_position'].shape} vs "
                f"{data['action_joint_position'].shape}"
            )

        # -----------------------------
        # Basic active_cube sanity check
        # -----------------------------
        active_cube = data["active_cube"]

        if np.any(active_cube < 0):
            episode_errors.append(
                "active_cube contains negative index"
            )

        # -----------------------------
        # Timestamp sanity
        # -----------------------------
        if len(timestamp) > 1:
            time_diff = np.diff(timestamp)

            if np.any(time_diff < 0):
                episode_errors.append(
                    "timestamp is not monotonically increasing"
                )

        if episode_errors:
            invalid_episodes.append(
                (episode_dir.name, episode_errors)
            )

    # =====================================
    # Summary
    # =====================================

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total = len(episode_dirs)

    print(f"Total episodes : {total}")
    print(f"Success        : {success_count}")
    print(f"Failure        : {failure_count}")

    if total > 0:
        success_rate = success_count / total * 100.0
        print(f"Success rate   : {success_rate:.2f}%")

    if frame_counts:
        print()
        print("Frame statistics:")
        print(f"  Average : {np.mean(frame_counts):.1f}")
        print(f"  Min     : {np.min(frame_counts)}")
        print(f"  Max     : {np.max(frame_counts)}")

    # =====================================
    # Failure reason distribution
    # =====================================

    print()
    print("Failure reasons:")

    if failure_reasons:
        for reason, count in failure_reasons.most_common():
            print(f"  {count:4d}  {reason}")
    else:
        print("  None")

    # =====================================
    # Invalid episodes
    # =====================================

    print()
    print("=" * 60)
    print("Dataset integrity")
    print("=" * 60)

    if not invalid_episodes:
        print("PASS: no structural problems found.")
    else:
        print(
            f"FAIL: {len(invalid_episodes)} invalid episodes found."
        )

        for episode_name, errors in invalid_episodes:
            print()
            print(episode_name)

            for error in errors:
                print(f"  - {error}")


if __name__ == "__main__":
    main()