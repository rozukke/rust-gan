use pyo3::{prelude::*, types::PyList};
use std::fs;
use std::path::Path;

fn main() -> PyResult<()> {
    let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/pysrc"));
    let py_app = fs::read_to_string(path.join("main.py"))?;
    let from_python = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        let syspath = py
            .import_bound("sys")?
            .getattr("path")?
            .downcast_into::<PyList>()?;
        syspath.insert(0, path)?;
        let app: Py<PyAny> = PyModule::from_code_bound(py, &py_app, "main.py", "main")?
            .getattr("main")?
            .into();
        app.call0(py)
    })?;

    println!("py: {}", from_python);
    Ok(())
}
