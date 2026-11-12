// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Video frame extraction API for Lance blob v2.
//!
//! This module provides demo APIs for extracting video frames from blob columns
//! that contain GOP (Group of Pictures) structured video data with blob_info metadata.

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt8Type, UInt64Type};
use arrow_array::{Array, StructArray};
use lance_core::Result;
use lance_core::datatypes::BlobKind;
use object_store::path::Path;
use object_store::{ObjectStoreExt, parse_url};
use serde_json::Value;
use url::Url;

use crate::dataset::{Dataset, ProjectionRequest};

#[derive(Debug, Clone)]
pub struct VideoFrameBytes {
    pub bytes: Vec<u8>,
    pub frame_index: u32,
    pub gop_id: u32,
    pub frame_type: u8,
    pub pts: u64,
}

fn json_u64(value: &Value, key: &str) -> Result<u64> {
    value.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        lance_core::Error::invalid_input(format!(
            "missing or invalid \"{}\" in blob_info json",
            key
        ))
    })
}

/// Returns the parsed URL when `blob_uri` is an absolute URI with a real
/// scheme. Windows drive-letter paths (`C:\...`) parse as single-letter URL
/// schemes and must be treated as filesystem paths instead.
fn absolute_blob_url(blob_uri: &str) -> Option<Url> {
    let url = Url::parse(blob_uri).ok()?;
    (url.scheme().len() > 1).then_some(url)
}

pub async fn take_frame_by_index(
    dataset: &Dataset,
    column: &str,
    row_id: u64,
    frame_index: u32,
) -> Result<VideoFrameBytes> {
    let projection = ProjectionRequest::from_columns(vec![column], dataset.schema());
    let record = dataset.take(&[row_id], projection).await?;

    if record.num_rows() == 0 {
        return Err(lance_core::Error::invalid_input(format!(
            "row_id {} not found in dataset",
            row_id
        )));
    }

    let struct_arr = record
        .column_by_name(column)
        .ok_or_else(|| lance_core::Error::invalid_input(format!("column {} not found", column)))?;
    let blob_desc = struct_arr
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            lance_core::Error::invalid_input(format!(
                "column {} is not a blob v2 descriptor struct",
                column
            ))
        })?;
    if blob_desc.is_null(0) {
        return Err(lance_core::Error::invalid_input(format!(
            "column {} value at row_id {} is null",
            column, row_id
        )));
    }

    let kind_arr = blob_desc
        .column_by_name("kind")
        .ok_or_else(|| lance_core::Error::invalid_input("kind not found in descriptor"))?;
    if kind_arr.is_null(0) {
        return Err(lance_core::Error::invalid_input("kind is null"));
    }
    let kind = BlobKind::try_from(kind_arr.as_primitive::<UInt8Type>().value(0))?;
    if kind != BlobKind::External {
        return Err(lance_core::Error::invalid_input(format!(
            "blob kind {:?} does not support frame extraction: expected External",
            kind
        )));
    }

    let position_arr = blob_desc
        .column_by_name("position")
        .ok_or_else(|| lance_core::Error::invalid_input("position not found in descriptor"))?;
    if position_arr.is_null(0) {
        return Err(lance_core::Error::invalid_input("position is null"));
    }
    let position = position_arr.as_primitive::<UInt64Type>().value(0);

    let blob_uri_arr = blob_desc
        .column_by_name("blob_uri")
        .ok_or_else(|| lance_core::Error::invalid_input("blob_uri not found in descriptor"))?;
    if blob_uri_arr.is_null(0) {
        return Err(lance_core::Error::invalid_input("blob_uri is null"));
    }
    let blob_uri = blob_uri_arr.as_string::<i32>().value(0);
    if blob_uri.is_empty() {
        return Err(lance_core::Error::invalid_input(
            "blob_uri is empty for external blob",
        ));
    }

    let blob_info = blob_desc.column_by_name("blob_info").ok_or_else(|| {
        lance_core::Error::invalid_input("blob_info field not found in descriptor")
    })?;
    let blob_info_binary = blob_info.as_binary::<i64>();
    if blob_info_binary.is_null(0) {
        return Err(lance_core::Error::invalid_input("blob_info value is null"));
    }
    let blob_info_bytes = blob_info_binary.value(0);
    let blob_info_value: Value = serde_json::from_slice(blob_info_bytes).map_err(|e| {
        lance_core::Error::invalid_input(format!("failed to parse blob_info json: {}", e))
    })?;

    let entries = blob_info_value
        .get("entries")
        .ok_or_else(|| lance_core::Error::invalid_input("entries field not found in blob_info"))?;

    let frames = entries
        .get("frames")
        .and_then(|f| f.as_array())
        .ok_or_else(|| lance_core::Error::invalid_input("frames field not found in entries"))?;
    let frame = frames.get(frame_index as usize).ok_or_else(|| {
        lance_core::Error::invalid_input(format!(
            "frame index {} out of range ({} frames)",
            frame_index,
            frames.len()
        ))
    })?;
    let frame_byte_offset = json_u64(frame, "byte_offset")?;
    let frame_byte_length = json_u64(frame, "byte_length")?;
    let frame_gop_id = json_u64(frame, "gop_id")? as u32;
    let frame_frame_type = json_u64(frame, "frame_type")? as u8;
    let frame_pts = json_u64(frame, "pts")?;

    let gops = entries
        .get("gops")
        .and_then(|g| g.as_array())
        .ok_or_else(|| lance_core::Error::invalid_input("gops field not found in entries"))?;
    let gop = gops.get(frame_gop_id as usize).ok_or_else(|| {
        lance_core::Error::invalid_input(format!(
            "gop id {} out of range ({} gops)",
            frame_gop_id,
            gops.len()
        ))
    })?;
    // Validate the gop entry shape even though the frame offset is absolute.
    json_u64(gop, "byte_offset")?;

    // Frame byte offsets in blob_info.entries are absolute within the blob
    // payload (recorded by prepare_video_demo.py from the file position when
    // writing), so only `position` (the blob's start offset in the object)
    // needs to be added. Adding the gop offset again would double-count it.
    let abs_start = position
        .checked_add(frame_byte_offset)
        .ok_or_else(|| lance_core::Error::invalid_input("frame byte offset overflow"))?;
    let abs_end = abs_start
        .checked_add(frame_byte_length)
        .ok_or_else(|| lance_core::Error::invalid_input("frame byte length overflow"))?;

    let bytes = if let Some(url) = absolute_blob_url(blob_uri) {
        // blob_uri is an absolute external URI (e.g. s3:// or file://):
        // connect to its object store directly instead of the dataset's own.
        let (store, blob_path) = parse_url(&url).map_err(|e| {
            lance_core::Error::invalid_input(format!("invalid blob_uri {:?}: {}", blob_uri, e))
        })?;
        store.get_range(&blob_path, abs_start..abs_end).await?
    } else {
        // blob_uri is a path relative to the dataset (or a local absolute path
        // without a URI scheme); resolve it against the dataset's object store.
        let object_store = dataset.object_store(None).await?;
        let path = Path::from(blob_uri);
        object_store
            .read_one_range(&path, abs_start as usize..abs_end as usize)
            .await?
    };

    Ok(VideoFrameBytes {
        bytes: bytes.to_vec(),
        frame_index,
        gop_id: frame_gop_id,
        frame_type: frame_frame_type,
        pts: frame_pts,
    })
}
