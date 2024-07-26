{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system;
          config = {
            inherit system overlays;
            allowUnfree = true;
            cudatoolkit = pkgs.cudatoolkit_12;
          };
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            rust-bin.stable.latest.default
            python3
            python3Packages.torch # WithCuda
            python3Packages.torchvision
            python3Packages.opencv4
            # cudatoolkit_12
          ];

          # shellHook = ''
            # export CUDA_HOME=${pkgs.cudatoolkit_12}
            # export PATH=$CUDA_HOME/bin:$PATH
            # export LD_LIBRARY_PATH=${pkgs.cudatoolkit_12}/lib64:$LD_LIBRARY_PATH
          shellHook = ''
            echo "Checking PyTorch + CUDA..."
            python3 -c 'import torch; print("CUDA found." if torch.cuda.is_available() else "CUDA not found.")'
          '';
        };
      }
    );
}