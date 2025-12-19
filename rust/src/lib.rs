use pyo3::prelude::*;
use numpy::{PyArray1};

mod transforms;

#[pyclass]
pub struct PyTransform {
    inner: transforms::Transform,
}

#[pymethods]
impl PyTransform {
    #[new]
    fn new() -> Self {
        PyTransform {
            inner: transforms::Transform::new(),
        }
    }
    
    fn fft_forward(&mut self, py: Python, signal: Vec<f32>) -> (Py<PyArray1<f32>>, Py<PyArray1<f32>>) {
        let result = self.inner.fft_forward(&signal);
        let re: Vec<f32> = result.iter().map(|c| c.re).collect();
        let im: Vec<f32> = result.iter().map(|c| c.im).collect();
        (
            PyArray1::from_vec(py, re).to_owned(),
            PyArray1::from_vec(py, im).to_owned()
        )
    }
    
    fn hartley_fast(&mut self, py: Python, signal: Vec<f32>) -> Py<PyArray1<f32>> {
        let result = self.inner.hartley_fast(&signal);
        PyArray1::from_vec(py, result).to_owned()
    }
}

#[pymodule]
fn nhfac_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTransform>()?;
    Ok(())
}
