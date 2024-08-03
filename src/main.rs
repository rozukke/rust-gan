use clap::{Parser, ValueEnum};
use pyo3::{prelude::*, types::PyList};
use std::fs;
use std::path::Path;
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

/// Get model from models folder without having to specify name
fn search_model(folder_path: &str, extension: &str) -> Option<std::path::PathBuf> {
    info!("Searching {} for model files...", folder_path);
    let path = Path::new(folder_path);
    for entry in fs::read_dir(path).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
            if let Some(path_str) = path.to_str() {
                info!("Found model file: {}", path_str);
            }
            return Some(path);
        }
    }
    None
}

fn main() -> Result<(), ()> {
    // Output logs to console
    tracing::subscriber::set_global_default(FmtSubscriber::builder().with_target(false).finish())
        .expect("Could not set stdout as logging output.");

    let args = Args::parse();
    let mut use_cpu = args.device == Device::Cpu;
    let input_path = args.input.unwrap_or(String::from(DEFAULT_INPUT));
    let output_path = args.output.unwrap_or(String::from(DEFAULT_OUTPUT));
    let model_path = match search_model(concat!(env!("CARGO_MANIFEST_DIR"), "/model"), "pth") {
        Some(model) => model.into_os_string(),
        _ => {
            error!("Could not find suitable model file in 'model' folder. Expected a '.pth' file extension.");
            return Err(());
        }
    };
    let half_precision = args.half_precision;

    // Import Python code from subdir
    let pysrc_path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/pysrc"));
    let infer_src = fs::read_to_string(pysrc_path.join("main.py")).map_err(|err| {
        error!("Could not open file for reading: {}", err);
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
