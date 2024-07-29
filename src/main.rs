use clap::{Parser, ValueEnum};
use pyo3::{prelude::*, types::PyList};
use std::fs;
use std::path::Path;
use tracing::{error, info, span, warn, Level};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Device to use for inference
    #[arg(value_enum)]
    device: Device,

    /// Path to input PNG image file to upscale
    #[arg(long)]
    input: Option<String>,

    /// Path to save output image file
    #[arg(long)]
    output: Option<String>,

    /// Use half precision for upscaling (faster, lower quality)
    #[arg(long, value_name = "half")]
    half_precision: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Device {
    /// Run on CPU
    Cpu,
    /// Run on GPU device of ID 0
    Gpu,
}

const DEFAULT_INPUT: &str = "./input/lowres.png";
const DEFAULT_OUTPUT: &str = "./input/highres.png";

fn main() -> PyResult<()> {
    // Output logs to console
    tracing::subscriber::set_global_default(tracing_subscriber::FmtSubscriber::new())
        .expect("Could not set stdout as logging output.");

    let args = Args::parse();
    let mut use_cpu = args.device == Device::Cpu;
    let _input_path = args.input.unwrap_or(String::from(DEFAULT_INPUT));
    let _output_path = args.output.unwrap_or(String::from(DEFAULT_OUTPUT));

    // Import Python code from subdir
    let py_path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/pysrc"));
    let main_src = fs::read_to_string(py_path.join("main.py"))?;

    let from_python = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        info!("Python initialised successfully.");
        {
            let _check_span = span!(Level::INFO, "ENV CHECKS").entered();

            info!("Checking PyTorch availability...");
            match Python::import_bound(py, "torch") {
                Ok(_) => info!("PyTorch is avaliable."),
                Err(err) => {
                    error!("PyTorch not found. Please ensure it is installed correctly.");
                    return Err(err);
                }
            }

            info!("Checking CUDA availability...");
            match Python::import_bound(py, "torch")?
                .getattr("cuda")?
                .getattr("is_available")?
                .call0()?
                .extract::<bool>()?
            {
                true => info!("CUDA is avaliable."),
                false => {
                    warn!("CUDA not found. Using CPU implementation.");
                    use_cpu = true;
                }
            }
        }

        // Set syspath for relative python imports
        let syspath = py
            .import_bound("sys")?
            .getattr("path")?
            .downcast_into::<PyList>()?;
        syspath.insert(0, py_path)?;

        // Run inference
        let app: Py<PyAny> = PyModule::from_code_bound(py, &main_src, "main.py", "main")?
            .getattr("main")?
            .into();
        app.call0(py)
    })?;

    info!("Result from python: {}", from_python);
    Ok(())
}
