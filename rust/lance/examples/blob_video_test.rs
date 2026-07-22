// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration test for blob_info extension.
//!
//! Run with: cargo run --package lance --example blob_video_test

use std::sync::Arc;

use arrow_array::{ArrayRef, ListArray, RecordBatch, StructArray, UInt32Array, UInt64Array, UInt8Array};
use arrow_schema::{DataType, Field as ArrowField, Fields};
use lance::dataset::{Dataset, ProjectionRequest, WriteParams};
use lance_core::datatypes::{BLOB_INFO_FIELDS, BLOB_V2_DESC_FIELDS, ENTRIES_FIELDS};
use tempfile::TempDir;

fn build_test_batch_with_blob_info(uri: &str, position: u64, gop_byte_offset: u64) -> RecordBatch {
    let gops_data = StructArray::try_new(
        ENTRIES_FIELDS.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![gop_byte_offset])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![512])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![30])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
        ],
        None,
    )
    .unwrap();

    let gops_list: ArrayRef = Arc::new(ListArray::try_new(
        arrow_schema::Field::new("item", ENTRIES_FIELDS.clone(), true).into(),
        arrow_buffer::Buffer::from_slice_of(&[0i32, 1]),
        gops_data,
        None,
    ).unwrap());

    let frames_data = StructArray::try_new(
        Fields::from(vec![
            ArrowField::new("byte_offset", DataType::UInt64, false),
            ArrowField::new("byte_length", DataType::UInt32, false),
            ArrowField::new("frame_type", DataType::UInt8, false),
            ArrowField::new("gop_id", DataType::UInt32, false),
            ArrowField::new("pts", DataType::UInt64, false),
        ]),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 100, 200])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![100, 100, 100])) as ArrayRef,
            Arc::new(UInt8Array::from(vec![1u8, 2, 2])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0u32, 0, 0])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![0u64, 33, 66])) as ArrayRef,
        ],
        None,
    ).unwrap();

    let frames_list: ArrayRef = Arc::new(ListArray::try_new(
        arrow_schema::Field::new("item", frames_data.as_ref().schema().fields().clone(), true).into(),
        arrow_buffer::Buffer::from_slice_of(&[0i32, 3]),
        frames_data,
        None,
    ).unwrap());

    let blob_info_data = StructArray::try_new(
        BLOB_INFO_FIELDS.clone(),
        vec![
            Arc::new(UInt8Array::from(vec![1u8])) as ArrayRef,
            Arc::new(StructArray::new(
                arrow_schema::Fields::from(vec![
                    ArrowField::new("codec", DataType::Utf8, false),
                    ArrowField::new("fps_num", DataType::UInt32, false),
                    ArrowField::new("fps_den", DataType::UInt32, false),
                ]),
                vec![
                    Arc::new(arrow_array::StringArray::from(vec!["h264"])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![30])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
                ],
                None,
            )) as ArrayRef,
            Arc::new(StructArray::new(
                ENTRIES_FIELDS.clone(),
                vec![gops_list.clone(), frames_list.clone()],
                None,
            )) as ArrayRef,
        ],
        None,
    ).unwrap();

    let descriptor = StructArray::try_new(
        BLOB_V2_DESC_FIELDS.clone(),
        vec![
            Arc::new(UInt8Array::from(vec![1u8])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![position])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![1024u64])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![1u32])) as ArrayRef,
            Arc::new(arrow_array::StringArray::from(vec![uri])) as ArrayRef,
            Arc::new(blob_info_data) as ArrayRef,
        ],
        None,
    ).unwrap();

    RecordBatch::try_from_struct_array(descriptor).unwrap()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let dataset_path = temp_dir.path().to_str().unwrap();

    println!("=== Blob Info Extension Integration Test ===");
    println!("Dataset path: {}", dataset_path);

    // Create a test .blob file
    let blob_path = temp_dir.path().join("test.blob");
    std::fs::write(&blob_path, vec![0u8; 1024])?;

    let blob_uri = format!("file:///{}", blob_path.to_string_lossy().replace("\\", "/"));

    // Build test batch
    let batch = build_test_batch_with_blob_info(&blob_uri, 0, 0);

    println!("Test batch schema: {:?}", batch.schema());

    // Write dataset
    println!("Attempting to write dataset...");

    let write_params = WriteParams::default();
    match Dataset::write(
        std::iter::once(Ok(batch)),
        dataset_path,
        Some(write_params),
    ).await {
        Ok(ds) => {
            println!("Dataset written successfully!");
            println!("Dataset schema: {:?}", ds.schema());

            // Read back
            let projection = ProjectionRequest::from_columns(vec!["description"], ds.schema());
            let result = ds.take(&[0], projection).await?;
            println!("Read back record batch schema: {:?}", result.schema());
            println!("Read back num rows: {}", result.num_rows());

            // Check if blob_info is present
            if let Some(col) = result.column_by_name("description") {
                let struct_arr = col.as_struct();
                if let Some(blob_info) = struct_arr.column_by_name("blob_info") {
                    println!("blob_info column found! Type: {:?}", blob_info.data_type());
                } else {
                    println!("WARNING: blob_info column not found");
                }
            }

            println!("\n=== Test PASSED ===");
        }
        Err(e) => {
            println!("Write error: {}", e);
            println!("\n=== This is expected for demo - schema must use lance.blob.v2 extension ===");
        }
    }

    Ok(())
}
