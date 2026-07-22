// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Video frame extraction API for Lance blob v2.
//!
//! This module provides demo APIs for extracting video frames from blob columns
//! that contain GOP (Group of Pictures) structured video data with blob_info metadata.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use lance_core::Result;
use object_store::path::Path;

use crate::dataset::{Dataset, ProjectionRequest};

#[derive(Debug, Clone)]
pub struct VideoFrameBytes {
    pub bytes: Vec<u8>,
    pub frame_index: u32,
    pub gop_id: u32,
    pub frame_type: u8,
    pub pts: u64,
}

fn extract_frame_index(array: &dyn Array, index: usize) -> Result<u32> {
    let val = array
        .as_primitive::<arrow_array::types::UInt32Type>()
        .value(index);
    Ok(val)
}

fn extract_u64(array: &dyn Array, index: usize) -> Result<u64> {
    let val = array
        .as_primitive::<arrow_array::types::UInt64Type>()
        .value(index);
    Ok(val)
}

fn extract_u8(array: &dyn Array, index: usize) -> Result<u8> {
    let val = array
        .as_primitive::<arrow_array::types::UInt8Type>()
        .value(index);
    Ok(val)
}

pub async fn take_frame_by_index(
    dataset: &Dataset,
    column: &str,
    row_id: u64,
    frame_index: u32,
) -> Result<VideoFrameBytes> {
    let projection = ProjectionRequest::from_columns(vec![column], dataset.schema());
    let record = dataset.take(&[row_id], projection).await?;

    let struct_arr = record
        .column_by_name(column)
        .ok_or_else(|| lance_core::Error::invalid_input(format!("column {} not found", column)))?;

    let blob_desc = struct_arr.as_struct();

    let position_arr = blob_desc
        .column_by_name("position")
        .ok_or_else(|| lance_core::Error::invalid_input("position not found in descriptor"))?;
    let position = extract_u64(position_arr.as_ref(), 0)?;

    let blob_uri_arr = blob_desc
        .column_by_name("blob_uri")
        .ok_or_else(|| lance_core::Error::invalid_input("blob_uri not found in descriptor"))?;
    let blob_uri = blob_uri_arr.as_string::<i32>().value(0).to_string();

    let blob_info = blob_desc.column_by_name("blob_info").ok_or_else(|| {
        lance_core::Error::invalid_input("blob_info field not found in descriptor")
    })?;

    let blob_info_struct = blob_info.as_struct();

    let entries = blob_info_struct
        .column_by_name("entries")
        .ok_or_else(|| lance_core::Error::invalid_input("entries field not found in blob_info"))?;

    let entries_struct = entries.as_struct();

    let frames_list = entries_struct
        .column_by_name("frames")
        .ok_or_else(|| lance_core::Error::invalid_input("frames field not found in entries"))?;

    let frames_list_arr = frames_list
        .as_list_opt::<i64>()
        .ok_or_else(|| lance_core::Error::invalid_input("frames is not a LargeList"))?;
    let frames_values = frames_list_arr.values();
    let frames_struct = frames_values.as_struct();

    let byte_offset_arr = frames_struct
        .column_by_name("byte_offset")
        .ok_or_else(|| lance_core::Error::invalid_input("byte_offset not found in frames"))?;
    let byte_length_arr = frames_struct
        .column_by_name("byte_length")
        .ok_or_else(|| lance_core::Error::invalid_input("byte_length not found in frames"))?;
    let gop_id_arr = frames_struct
        .column_by_name("gop_id")
        .ok_or_else(|| lance_core::Error::invalid_input("gop_id not found in frames"))?;
    let frame_type_arr = frames_struct
        .column_by_name("frame_type")
        .ok_or_else(|| lance_core::Error::invalid_input("frame_type not found in frames"))?;
    let pts_arr = frames_struct
        .column_by_name("pts")
        .ok_or_else(|| lance_core::Error::invalid_input("pts not found in frames"))?;

    let frame_byte_offset = extract_u64(byte_offset_arr.as_ref(), frame_index as usize)?;
    let frame_byte_length = extract_u64(byte_length_arr.as_ref(), frame_index as usize)? as u64;
    let frame_gop_id = extract_frame_index(gop_id_arr.as_ref(), frame_index as usize)?;
    let frame_frame_type = extract_u8(frame_type_arr.as_ref(), frame_index as usize)?;
    let frame_pts = extract_u64(pts_arr.as_ref(), frame_index as usize)?;

    let gops_list = entries_struct
        .column_by_name("gops")
        .ok_or_else(|| lance_core::Error::invalid_input("gops field not found in entries"))?;

    let gop_list_arr = gops_list
        .as_list_opt::<i64>()
        .ok_or_else(|| lance_core::Error::invalid_input("gops is not a LargeList"))?;
    let gop_values = gop_list_arr.values();
    let gop_struct = gop_values.as_struct();

    let gop_offset_arr = gop_struct
        .column_by_name("byte_offset")
        .ok_or_else(|| lance_core::Error::invalid_input("byte_offset not found in gops"))?;
    let gop_byte_offset = extract_u64(gop_offset_arr.as_ref(), frame_gop_id as usize)?;

    let abs_start = position + gop_byte_offset;
    let abs_end = abs_start + frame_byte_offset + frame_byte_length;

    let object_store = dataset.object_store(None).await?;
    let path = Path::from(blob_uri);
    let bytes = object_store
        .read_one_range(&path, abs_start as usize..abs_end as usize)
        .await?;

    Ok(VideoFrameBytes {
        bytes: bytes.to_vec(),
        frame_index,
        gop_id: frame_gop_id,
        frame_type: frame_frame_type,
        pts: frame_pts,
    })
}
