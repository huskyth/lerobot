#!/usr/bin/env python
"""Repair a HIL-SERL replay dataset interrupted while writing parquet files.

The repair keeps the longest valid prefix of data shards, rebuilds minimal v3
episode metadata from the surviving rows, validates a staged copy, and only
then updates the dataset. The original damaged files and metadata are moved to
a timestamped backup directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EPISODE_COLUMNS = [
    "index",
    "episode_index",
    "frame_index",
    "task_index",
    "next.done",
]


def atomic_write_json(path: Path, value: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".repair.tmp")
    tmp_path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def inspect_valid_prefix(root: Path) -> tuple[list[tuple[int, Path, pd.DataFrame]], list[Path]]:
    data_dir = root / "data" / "chunk-000"
    files = sorted(data_dir.glob("file-*.parquet"))
    valid: list[tuple[int, Path, pd.DataFrame]] = []
    invalid: list[Path] = []

    for file_index, path in enumerate(files):
        try:
            table = pq.read_table(path, columns=EPISODE_COLUMNS)
        except Exception:
            invalid.append(path)
            break
        valid.append((file_index, path, table.to_pandas()))

    if not valid:
        raise RuntimeError(f"No readable data shards found in {data_dir}")

    all_rows = pd.concat([frame for _, _, frame in valid], ignore_index=True)
    expected_indices = range(len(all_rows))
    if not all_rows["index"].eq(list(expected_indices)).all():
        raise RuntimeError("Data shard indices are not contiguous; refusing automatic repair")

    for _, path, frame in valid:
        last_episode = int(frame["episode_index"].iloc[-1])
        if last_episode >= 0 and not bool(frame["next.done"].iloc[-1]):
            raise RuntimeError(f"The final episode in {path} is incomplete")

    return valid, invalid


def build_episode_table(
    valid_files: list[tuple[int, Path, pd.DataFrame]], task_name: str
) -> pa.Table:
    episodes: list[dict] = []
    expected_episode = 0

    for file_index, path, frame in valid_files:
        for episode_index, rows in frame.groupby("episode_index", sort=True):
            episode_index = int(episode_index)
            if episode_index != expected_episode:
                raise RuntimeError(
                    f"Expected episode {expected_episode}, found {episode_index} in {path}"
                )
            if len({int(value) for value in rows["task_index"]}) != 1:
                raise RuntimeError(f"Episode {episode_index} contains multiple task indices")
            frames = rows["frame_index"].to_numpy()
            if not (frames == range(len(frames))).all():
                raise RuntimeError(f"Episode {episode_index} has non-contiguous local frames")

            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task_name],
                    "length": len(rows),
                    "data/chunk_index": 0,
                    "data/file_index": file_index,
                    "dataset_from_index": int(rows["index"].iloc[0]),
                    "dataset_to_index": int(rows["index"].iloc[-1]) + 1,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            expected_episode += 1

    return pa.Table.from_pylist(
        episodes,
        schema=pa.schema(
            [
                pa.field("episode_index", pa.int64()),
                pa.field("tasks", pa.list_(pa.string())),
                pa.field("length", pa.int64()),
                pa.field("data/chunk_index", pa.int64()),
                pa.field("data/file_index", pa.int64()),
                pa.field("dataset_from_index", pa.int64()),
                pa.field("dataset_to_index", pa.int64()),
                pa.field("meta/episodes/chunk_index", pa.int64()),
                pa.field("meta/episodes/file_index", pa.int64()),
            ]
        ),
    )


def task_name_from_metadata(root: Path) -> str:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks.index) != 1:
        raise RuntimeError("Expected exactly one named task in meta/tasks.parquet")
    return str(tasks.index[0])


def stage_and_validate(
    root: Path,
    valid_files: list[tuple[int, Path, pd.DataFrame]],
    episode_table: pa.Table,
    info: dict,
) -> None:
    from lerobot.datasets import LeRobotDataset

    with tempfile.TemporaryDirectory(prefix="lerobot-dataset-repair-") as temp_dir:
        stage = Path(temp_dir)
        (stage / "meta").mkdir(parents=True)
        (stage / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (stage / "data" / "chunk-000").mkdir(parents=True)

        for name in ("tasks.parquet", "stats.json"):
            shutil.copy2(root / "meta" / name, stage / "meta" / name)
        atomic_write_json(stage / "meta" / "info.json", info)
        pq.write_table(
            episode_table,
            stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )

        for _, source, _ in valid_files:
            (stage / "data" / "chunk-000" / source.name).symlink_to(source)
        for image_dir in (root / "images").iterdir():
            (stage / "images").mkdir(exist_ok=True)
            (stage / "images" / image_dir.name).symlink_to(image_dir)

        dataset = LeRobotDataset(repo_id="local/replay-repair-validation", root=stage)
        if len(dataset) != int(info["total_frames"]):
            raise RuntimeError(f"Staged dataset length {len(dataset)} != {info['total_frames']}")
        dataset[0]
        dataset[len(dataset) - 1]
        print(
            f"Validated staged replay dataset: {len(dataset)} frames, "
            f"{int(info['total_episodes'])} episodes"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="HIL-SERL output dataset directory containing data/ and meta/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated repair; without this flag only inspect and stage",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    valid_files, invalid_files = inspect_valid_prefix(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task_name = task_name_from_metadata(root)
    episode_table = build_episode_table(valid_files, task_name)

    total_frames = sum(len(frame) for _, _, frame in valid_files)
    info["total_episodes"] = episode_table.num_rows
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{episode_table.num_rows}"}
    info["total_tasks"] = 1

    print(
        f"Readable prefix: {len(valid_files)} data shards, {total_frames} frames, "
        f"{episode_table.num_rows} episodes"
    )
    for path in invalid_files:
        print(f"Damaged shard: {path.relative_to(root)}")

    stage_and_validate(root, valid_files, episode_table, info)
    if not args.apply:
        print("Dry run complete. Re-run with --apply to update the dataset.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root.parent / f"{root.name}_repair_backup_{stamp}"
    backup.mkdir(parents=True)

    damaged_metadata = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    to_backup = [*invalid_files]
    if damaged_metadata.exists():
        to_backup.append(damaged_metadata)
    for source in to_backup:
        destination = backup / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
    shutil.copy2(info_path, backup / "info.json.before_repair")

    episode_path = damaged_metadata
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_table, episode_path)
    atomic_write_json(info_path, info)

    print(f"Repair applied. Original damaged files are in {backup}")


if __name__ == "__main__":
    main()
