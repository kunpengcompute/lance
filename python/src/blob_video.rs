// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Borrow;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use lance::VideoFrameBytes as LanceVideoFrameBytes;
use lance::take_frame_by_index as take_frame_by_index_impl;

use crate::{Dataset as PyDataset, rt};

#[pyclass(name = "VideoFrameBytes")]
pub struct PyVideoFrameBytes {
    pub bytes: Vec<u8>,
    pub frame_index: u32,
    pub gop_id: u32,
    pub frame_type: u8,
    pub pts: u64,
}

#[pymethods]
impl PyVideoFrameBytes {
    #[getter]
    fn bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    #[getter]
    fn frame_index(&self) -> u32 {
        self.frame_index
    }

    #[getter]
    fn gop_id(&self) -> u32 {
        self.gop_id
    }

    #[getter]
    fn frame_type(&self) -> u8 {
        self.frame_type
    }

    #[getter]
    fn pts(&self) -> u64 {
        self.pts
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoFrameBytes(frame_index={}, gop_id={}, frame_type={}, pts={}, size={})",
            self.frame_index,
            self.gop_id,
            self.frame_type,
            self.pts,
            self.bytes.len()
        )
    }
}

#[pyfunction(name = "take_frame_by_index")]
pub fn py_take_frame_by_index(
    dataset: &Bound<'_, PyAny>,
    column: PyBackedStr,
    row_id: u64,
    frame_index: u32,
) -> PyResult<PyVideoFrameBytes> {
    let py = dataset.py();
    let ds = dataset.getattr("_ds")?.extract::<Py<PyDataset>>()?;
    let ds_ref = &ds.bind(py).borrow().ds;

    let rt = rt();
    let result = rt.block_on(Some(py), async {
        take_frame_by_index_impl(ds_ref, &column, row_id, frame_index)
            .await
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("take_frame_by_index failed: {}", e))
            })
    });

    match result {
        Ok(Ok(frame)) => Ok(PyVideoFrameBytes {
            bytes: frame.bytes,
            frame_index: frame.frame_index,
            gop_id: frame.gop_id,
            frame_type: frame.frame_type,
            pts: frame.pts,
        }),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(e),
    }
}
