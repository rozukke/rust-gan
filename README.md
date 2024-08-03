# rust-gan
[![GitHub license](https://img.shields.io/github/license/rozukke/rust-gan.svg)](https://github.com/rozukke/rust-gan/blob/main/LICENSE)
[![Creator rozukke](https://img.shields.io/badge/Creator-rozukke-f497af.svg)](https://github.com/rozukke)
[![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-b7410e.svg)](https://www.rust-lang.org)

This is a repository meant to demonstrate running a PyTorch script from a Rust binary with versioning and deployment using Nix.

## Installation
Tested working on WSL with `Ubuntu-24.04 LTS`. Please ensure that you have the latest version Nvidia driver installed on your
system (tested with driver version `560.70`). Starting a shell with `nix develop` should print "CUDA found!" to the console.

## Usage
For development purposes, use `nix develop` to start a dev shell with all required packages. Place an appropriate model file into the `model` directory.
Ensure it has a `.pth` extension. Some models may not be able to be loaded, so a working one is provided here (TODO).

### Inference with shell
- The executable is in the format `rust-gan [OPTIONS] <DEVICE>`, where the device is either CPU or GPU.
- Provide a path to your image using `--input` and a path to save to with `--output`. Alternatively, run without arguments
to use the default input provided (found in `input/lowres.png`, output saved to `input/highres.png`).
- Use half precision with the flag `--half-precision`. Will be slower on CPU than full precision.

### Using Nix package
- TODO