use clap::{Parser, ValueEnum};
use pyo3::{prelude::*, types::PyList};
use std::path::Path;
use std::{fs, path::PathBuf};
use tracing::{error, info, span, warn, Level};
use tracing_subscriber::FmtSubscriber;

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

    /// Path to model file
    #[arg(long)]
    model: String,

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

fn main() -> Result<(), ()> {
    // Output logs to console
    tracing::subscriber::set_global_default(FmtSubscriber::builder().with_target(false).finish())
        .expect("Could not set stdout as logging output.");

    let args = Args::parse();
    let mut use_cpu = args.device == Device::Cpu;
    let input_path = args.input.unwrap_or(String::from(DEFAULT_INPUT));
    let output_path = args.output.unwrap_or(String::from(DEFAULT_OUTPUT));
    let model_path = args.model;
    if !Path::new(model_path.as_str()).exists() {
        error!("Could not find model file. Exiting...");
        return Err(());
    }
    let half_precision = args.half_precision;

    let pysrc_path = match std::env::var("PY_GAN_PATH") {
        Ok(pypath) => {
            info!("Found python sources at {}", pypath);
            PathBuf::from(pypath)
        },
        Err(_) => {
            error!("Could not find ENV location for Python sources.");
            return Err(());
        }
    };

    let infer_src = fs::read_to_string(pysrc_path.join("main.py")).map_err(|err| {
        error!(
            "Could not open {} for reading: {}",
            pysrc_path.display(),
            err
        );
    })?;

    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
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

        let device_str = if use_cpu { "cpu" } else { "cuda:0" };

        // Set syspath for relative python imports
        let syspath = py
            .import_bound("sys")?
            .getattr("path")?
            .downcast_into::<PyList>()?;
        syspath.insert(0, pysrc_path)?;

        // Run inference
        let app: Py<PyAny> = PyModule::from_code_bound(py, &infer_src, "main.py", "main")?
            .getattr("infer")?
            .into();
        info!("Running inference...");
        app.call1(
            py,
            (
                input_path,
                &output_path,
                model_path,
                device_str,
                half_precision,
            ),
        )
    })
    .map_err(|err| {
        error!("Python execution failed with error {}", err);
    })?;

    let _success = span!(Level::INFO, "SUCCESS").entered();
    info!(
        "Inference ran successfully. Image saved to path {}",
        output_path
    );
    Ok(())
}
