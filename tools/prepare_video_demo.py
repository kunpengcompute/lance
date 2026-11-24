#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Demo tool for preparing video data with GOP indexing for Lance blob columns.

This tool extracts GOP (Group of Pictures) structured video data and generates
the blob_info metadata needed for efficient frame extraction.

Usage:
    python prepare_video_demo.py input.mp4 output_dir [--row-id ROW_ID]

Output:
    - output_dir/video.blob: Raw GOP bytes
    - Returns a dict with blob descriptor for Lance write_dataset
"""

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import av
    import av.codec
    import av.container
    import av.stream
except ImportError:
    print("Error: PyAV is required. Install with: pip install av")
    sys.exit(1)

import pyarrow as pa


FRAME_TYPE_I = 1
FRAME_TYPE_P = 2
FRAME_TYPE_B = 3


@dataclass
class FrameInfo:
    byte_offset: int
    byte_length: int
    frame_type: int
    gop_id: int
    pts: int


@dataclass
class GopInfo:
    byte_offset: int
    byte_length: int
    frame_count: int
    first_frame_idx: int


@dataclass
class StreamInfo:
    codec: str
    fps_num: int
    fps_den: int
    duration_us: int
    width: int
    height: int
    extradata: Optional[bytes]


@dataclass
class BlobInfo:
    blob_type: int
    stream_info: StreamInfo
    gops: List[GopInfo]
    frames: List[FrameInfo]


def extract_stream_info(stream: av.stream.Stream) -> StreamInfo:
    """Extract stream metadata."""
    codec_ctx = stream.codec_context
    fps = stream.average_rate

    duration_us = 0
    if stream.duration is not None:
        duration_us = int(stream.duration * 1_000_000 / stream.time_base.denominator)
    elif stream.container.duration is not None:
        duration_us = int(
            stream.container.duration * 1_000_000 / stream.container.duration_time_base
        )

    extradata = None
    if codec_ctx.extradata is not None:
        extradata = bytes(codec_ctx.extradata)

    return StreamInfo(
        codec=codec_ctx.name,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        duration_us=duration_us,
        width=codec_ctx.width,
        height=codec_ctx.height,
        extradata=extradata,
    )


def prepare_video(input_mp4: str, output_blob_path: str) -> Tuple[BlobInfo, int]:
    """
    Prepare video data with GOP indexing.

    Parameters
    ----------
    input_mp4 : str
        Path to input MP4 file.
    output_blob_path : str
        Path to output .blob file.

    Returns
    -------
    Tuple[BlobInfo, int]
        BlobInfo metadata and total size in bytes.
    """
    container = av.open(input_mp4)
    stream = container.streams.video[0]

    stream_info = extract_stream_info(stream)

    gops: List[GopInfo] = []
    frames: List[FrameInfo] = []
    current_gop_id = -1
    current_gop_start_offset = 0
    current_gop_frame_count = 0
    current_gop_first_frame_idx = 0
    frame_idx = 0

    with open(output_blob_path, "wb") as f:
        for packet in container.demux(stream):
            if packet.dts is None:
                continue

            packet_bytes = bytes(packet)

            if packet.is_keyframe:
                if current_gop_id >= 0 and frame_idx > 0:
                    gops.append(
                        GopInfo(
                            byte_offset=current_gop_start_offset,
                            byte_length=f.tell() - current_gop_start_offset,
                            frame_count=current_gop_frame_count,
                            first_frame_idx=current_gop_first_frame_idx,
                        )
                    )
                    current_gop_first_frame_idx = frame_idx
                    current_gop_start_offset = f.tell()
                    current_gop_frame_count = 0

                current_gop_id += 1

            frame_offset = f.tell()
            f.write(packet_bytes)
            frame_length = len(packet_bytes)

            pict_type = getattr(packet, "pict_type", None)
            if pict_type is None:
                frame_type = FRAME_TYPE_P
            elif pict_type.value == 1:  # I frame
                frame_type = FRAME_TYPE_I
            elif pict_type.value == 2:  # P frame
                frame_type = FRAME_TYPE_P
            elif pict_type.value == 3:  # B frame
                frame_type = FRAME_TYPE_B
            else:
                frame_type = FRAME_TYPE_P

            frames.append(
                FrameInfo(
                    byte_offset=frame_offset,
                    byte_length=frame_length,
                    frame_type=frame_type,
                    gop_id=current_gop_id,
                    pts=int(packet.pts * 1_000_000 / packet.time_base.denominator)
                    if packet.pts
                    else 0,
                )
            )

            frame_idx += 1
            current_gop_frame_count += 1

        total_size = f.tell()

        if current_gop_id >= 0:
            gops.append(
                GopInfo(
                    byte_offset=current_gop_start_offset,
                    byte_length=total_size - current_gop_start_offset,
                    frame_count=current_gop_frame_count,
                    first_frame_idx=current_gop_first_frame_idx,
                )
            )

    blob_info = BlobInfo(
        blob_type=1,
        stream_info=stream_info,
        gops=gops,
        frames=frames,
    )

    return blob_info, total_size


def build_blob_descriptor(
    blob_uri: str,
    position: int,
    size: int,
    blob_id: int,
    blob_info: BlobInfo,
) -> Dict[str, Any]:
    """
    Build a BlobDescriptorV2 Arrow Struct dict for Lance.

    Parameters
    ----------
    blob_uri : str
        Absolute external URI to the .blob file (e.g. file:///...).
    position : int
        Byte offset in the blob file (usually 0 for single video).
    size : int
        Total size of the video in bytes.
    blob_id : int
        Blob ID. Must be 0 when blob_uri is an absolute external URI.
    blob_info : BlobInfo
        Video GOP indexing metadata.

    Returns
    -------
    Dict
        BlobDescriptorV2 dict whose blob_info is UTF-8 encoded JSON bytes,
        matching the large_binary storage contract on the Rust side.
    """
    gop_structs = []
    for gop in blob_info.gops:
        gop_structs.append(
            {
                "byte_offset": gop.byte_offset,
                "byte_length": gop.byte_length,
                "frame_count": gop.frame_count,
                "first_frame_idx": gop.first_frame_idx,
            }
        )

    frame_structs = []
    for frame in blob_info.frames:
        frame_structs.append(
            {
                "byte_offset": frame.byte_offset,
                "byte_length": frame.byte_length,
                "frame_type": frame.frame_type,
                "gop_id": frame.gop_id,
                "pts": frame.pts,
            }
        )

    stream_info_dict = {
        "codec": blob_info.stream_info.codec,
        "fps_num": blob_info.stream_info.fps_num,
        "fps_den": blob_info.stream_info.fps_den,
        "duration_us": blob_info.stream_info.duration_us,
        "width": blob_info.stream_info.width,
        "height": blob_info.stream_info.height,
        "extradata": base64.b64encode(blob_info.stream_info.extradata).decode()
        if blob_info.stream_info.extradata
        else None,
    }

    entries_dict = {
        "gops": gop_structs,
        "frames": frame_structs,
    }

    blob_info_dict = {
        "blob_type": blob_info.blob_type,
        "stream_info": stream_info_dict,
        "entries": entries_dict,
    }

    return {
        "kind": 3,  # External: not stored by Lance data files
        "position": position,
        "size": size,
        "blob_id": blob_id,
        "blob_uri": blob_uri,
        "blob_info": json.dumps(blob_info_dict).encode("utf-8"),
    }


def get_blob_v2_schema() -> pa.Schema:
    """Get Arrow schema for BlobDescriptorV2 with blob_info (JSON bytes)."""
    return pa.schema(
        [
            pa.field("kind", pa.uint8(), False),
            pa.field("position", pa.uint64(), False),
            pa.field("size", pa.uint64(), False),
            pa.field("blob_id", pa.uint32(), False),
            pa.field("blob_uri", pa.utf8(), False),
            pa.field("blob_info", pa.large_binary(), True),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video with GOP indexing for Lance blob columns"
    )
    parser.add_argument("input", help="Input MP4 file")
    parser.add_argument("output_dir", help="Output directory for .blob file")
    parser.add_argument("--blob-id", type=int, default=0, help="Blob ID (default: 0)")
    parser.add_argument("--row-id", type=int, default=0, help="Row ID (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    blob_path = os.path.join(args.output_dir, "video.blob")
    blob_uri = f"file:///{blob_path.replace(os.sep, '/')}"

    if args.verbose:
        print(f"Processing: {args.input}", file=sys.stderr)
        print(f"Output blob: {blob_path}", file=sys.stderr)

    blob_info, total_size = prepare_video(args.input, blob_path)

    descriptor = build_blob_descriptor(
        blob_uri=blob_uri,
        position=0,
        size=total_size,
        blob_id=args.blob_id,
        blob_info=blob_info,
    )

    if args.verbose:
        print(f"Total GOPs: {len(blob_info.gops)}", file=sys.stderr)
        print(f"Total frames: {len(blob_info.frames)}", file=sys.stderr)
        print(f"Total size: {total_size} bytes", file=sys.stderr)
        print(f"Codec: {blob_info.stream_info.codec}", file=sys.stderr)
        print(
            f"Resolution: {blob_info.stream_info.width}x{blob_info.stream_info.height}",
            file=sys.stderr,
        )
        print(
            f"FPS: {blob_info.stream_info.fps_num}/{blob_info.stream_info.fps_den}",
            file=sys.stderr,
        )

    result = {
        "descriptor": descriptor,
        "blob_path": blob_path,
        "blob_uri": blob_uri,
        "stats": {
            "num_gops": len(blob_info.gops),
            "num_frames": len(blob_info.frames),
            "total_size": total_size,
            "codec": blob_info.stream_info.codec,
            "fps": f"{blob_info.stream_info.fps_num}/{blob_info.stream_info.fps_den}",
            "resolution": f"{blob_info.stream_info.width}x{blob_info.stream_info.height}",
        },
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
