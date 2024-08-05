# rust-gan
[![GitHub license](https://img.shields.io/github/license/rozukke/rust-gan.svg)](https://github.com/rozukke/rust-gan/blob/main/LICENSE)
[![Creator rozukke](https://img.shields.io/badge/Creator-rozukke-f497af.svg)](https://github.com/rozukke)
[![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-b7410e.svg)](https://www.rust-lang.org)

This is a repository meant to demonstrate running a PyTorch script from a Rust binary with versioning and deployment using Nix.

## Installation
Tested working on WSL with `Ubuntu-24.04 LTS`. Please ensure that you have the latest version Nvidia driver installed on your
system (tested with driver version `560.70`). Starting a shell with `nix develop` should print "CUDA found!" to the console.

## Usage
For development purposes, use `nix develop` to start a dev shell with all required packages. Place an appropriate model file
into the `model` directory. One such model can be downloaded [here](https://drive.google.com/file/d/1fCKufxu-a0vewCP1Y_7DP_JmrPxBXYCF/view?usp=sharing).
It is recommended to use this model as a specific architecture is required.

### Inference with shell
- The executable is in the format `rust-gan [OPTIONS] <DEVICE>`, where the device is either `cpu` or `gpu`. If CUDA is not
accessible, the device will be overridden to use CPU with a warning.
- Provide a path to your image using `--input` and a path to save to with `--output`. Alternatively, run without arguments
to use the default input provided (found in `input/lowres.png`, output saved to `input/highres.png`).
- Provide a path to the model using a `--model` flag. Automatic model detection is not very compatible with nix packaging,
but might be added later.
- Use half precision with the flag `--half-precision`. Will be slower on CPU than full precision.

### Using Nix package
- It should be possible to run the nix package using `nix run ./ -- gpu --model /path/to/model`. The package is in a functional state, but isn't currently working 100% as might be expected.