use pyo3::prelude::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        PyModule::from_code_bound(
            py,
            r#"def hello():
                print("Hello from Python")
            "#,
            "",
            "",
        )?
        .getattr("hello")?
        .call0()?;
        Ok(())
    })
}
