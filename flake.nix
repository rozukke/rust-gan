{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaVersion = "12";
          };
        };
      in {
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            rust-bin.stable."1.80.0".default
            cudatoolkit
            python3
            python3Packages.torchWithCuda
            python3Packages.torchvision
            python3Packages.opencv4
          ];

          shellHook = ''
            # export CUDA_HOME=${pkgs.cudatoolkit}
            # export PATH=$CUDA_HOME/bin:$PATH
            # export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib64:$LD_LIBRARY_PATH
            echo "Checking PyTorch + CUDA..."
            python3 -c 'import torch; print("CUDA found." if torch.cuda.is_available() else "CUDA not found.")'
          '';
        };
      }
    );
}