# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
E2E tests for video frame extraction via blob_info metadata.

These tests verify:
1. GOP-split video can be written to Lance with blob_info metadata
2. take_frame_by_index returns correct frame bytes
3. Frame metadata (pts, frame_type) is correctly extracted
"""

import os
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import lance
from lance.blob import take_frame_by_index, BlobType, VideoFrameBytes


TEST_VIDEO_URL = "https://storage.googleapis.com/lance-artifacts/test-data/sample_video.mp4"
TEST_VIDEO_PATH = Path(__file__).parent.parent.parent / "test_data" / "sample_video.mp4"


@pytest.fixture(scope="module")
def test_video():
    """Download or locate test video file."""
    if TEST_VIDEO_PATH.exists():
        return str(TEST_VIDEO_PATH)

    print(f"Downloading test video from {TEST_VIDEO_URL}...")
    try:
        import urllib.request
        TEST_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEST_VIDEO_URL, str(TEST_VIDEO_PATH))
        return str(TEST_VIDEO_PATH)
    except Exception as e:
        pytest.skip(f"Failed to download test video: {e}")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def run_prepare_video(input_video: str, output_dir: str) -> dict:
    """Run the prepare_video_demo.py script."""
    script_path = Path(__file__).parent.parent.parent / "tools" / "prepare_video_demo.py"
    result = subprocess.run(
        [sys.executable, str(script_path), input_video, output_dir, "-v"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prepare_video_demo.py failed: {result.stderr}")
    import json
    return json.loads(result.stdout)


def get_blob_v2_ext_type() -> pa.ExtensionType:
    """Get BlobType for lance.blob.v2."""
    return BlobType()


def test_prepare_video_demo(test_video, temp_dir):
    """Test that prepare_video_demo.py correctly splits video into GOPs."""
    result = run_prepare_video(test_video, temp_dir)

    assert os.path.exists(result["blob_path"]), "Blob file should exist"
    assert result["stats"]["num_gops"] > 0, "Should have at least one GOP"
    assert result["stats"]["num_frames"] > 0, "Should have at least one frame"

    descriptor = result["descriptor"]
    assert descriptor["kind"] == 1, "Kind should be 1 (Packed)"
    assert descriptor["blob_info"]["blob_type"] == 1, "Blob type should be 1 (video)"
    assert "gops" in descriptor["blob_info"]["entries"]
    assert "frames" in descriptor["blob_info"]["entries"]


def test_write_dataset_with_blob_info(test_video, temp_dir):
    """Test writing a dataset with blob_info metadata."""
    result = run_prepare_video(test_video, temp_dir)
    descriptor = result["descriptor"]

    blob_type = get_blob_v2_ext_type()

    schema = pa.schema([
        pa.field("id", pa.int32(), False),
        pa.field("video", blob_type),
    ])

    table = pa.table({
        "id": [1],
        "video": [descriptor],
    })

    dataset_path = os.path.join(temp_dir, "video_dataset")
    lance.write_dataset(table, dataset_path, schema=schema)

    ds = lance.dataset(dataset_path)
    assert "video" in ds.schema.names
    assert "id" in ds.schema.names


def test_take_frame_by_index(test_video, temp_dir):
    """Test take_frame_by_index returns correct frame data."""
    result = run_prepare_video(test_video, temp_dir)
    descriptor = result["descriptor"]

    blob_type = get_blob_v2_ext_type()

    schema = pa.schema([
        pa.field("id", pa.int32(), False),
        pa.field("video", blob_type),
    ])

    table = pa.table({
        "id": [1],
        "video": [descriptor],
    })

    dataset_path = os.path.join(temp_dir, "video_dataset")
    lance.write_dataset(table, dataset_path, schema=schema)

    ds = lance.dataset(dataset_path)

    frame = take_frame_by_index(ds, "video", row_id=0, frame_index=0)

    assert isinstance(frame, VideoFrameBytes)
    assert frame.frame_index == 0
    assert frame.gop_id == 0
    assert frame.pts >= 0
    assert len(frame.bytes) > 0, "Frame bytes should not be empty"


def test_take_frame_by_index_i_frame(test_video, temp_dir):
    """Test that frame_index=0 returns an I-frame (keyframe)."""
    result = run_prepare_video(test_video, temp_dir)
    descriptor = result["descriptor"]

    blob_type = get_blob_v2_ext_type()

    schema = pa.schema([
        pa.field("id", pa.int32(), False),
        pa.field("video", blob_type),
    ])

    table = pa.table({
        "id": [1],
        "video": [descriptor],
    })

    dataset_path = os.path.join(temp_dir, "video_dataset")
    lance.write_dataset(table, dataset_path, schema=schema)

    ds = lance.dataset(dataset_path)

    frame = take_frame_by_index(ds, "video", row_id=0, frame_index=0)

    assert frame.frame_type == 1, "First frame should be an I-frame"


def test_metadata_correctness(test_video, temp_dir):
    """Test that extracted frame metadata matches blob_info."""
    result = run_prepare_video(test_video, temp_dir)
    descriptor = result["descriptor"]

    blob_type = get_blob_v2_ext_type()

    schema = pa.schema([
        pa.field("id", pa.int32(), False),
        pa.field("video", blob_type),
    ])

    table = pa.table({
        "id": [1],
        "video": [descriptor],
    })

    dataset_path = os.path.join(temp_dir, "video_dataset")
    lance.write_dataset(table, dataset_path, schema=schema)

    ds = lance.dataset(dataset_path)

    frame_idx = 10
    frame = take_frame_by_index(ds, "video", row_id=0, frame_index=frame_idx)

    expected_frame = descriptor["blob_info"]["entries"]["frames"][frame_idx]

    assert frame.frame_index == frame_idx
    assert frame.gop_id == expected_frame["gop_id"]
    assert frame.frame_type == expected_frame["frame_type"]
    assert frame.pts == expected_frame["pts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
