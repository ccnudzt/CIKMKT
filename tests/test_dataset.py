from pathlib import Path

import torch

from src.data.kt import filter_records_by_len, get_raw_records, kt_collate_fn, truncate_long_records


def test_records_pipeline(tmp_path: Path):
    records_path = tmp_path / "records.txt"
    records_path.write_text("1,2,3,4\n1,0,1,0\n5,6\n0,1\n", encoding="utf-8")

    raw = get_raw_records(records_path)
    truncated = truncate_long_records(raw, max_len=3)
    filtered = filter_records_by_len(truncated, min_len=2)

    assert len(raw) == 2
    assert len(truncated) == 3
    assert len(filtered) == 2


def test_collate_masks_padding():
    batch = [
        {"qids": torch.tensor([1, 2, 3]), "rs": torch.tensor([1, 0, 1])},
        {"qids": torch.tensor([4, 5]), "rs": torch.tensor([0, 1])},
    ]
    collated = kt_collate_fn(batch, max_len=4)
    assert tuple(collated["qids"].shape) == (2, 4)
    assert collated["masks"][1].tolist() == [1, 1, 0, 0]
    assert collated["rs"][1].tolist()[-2:] == [2, 2]
