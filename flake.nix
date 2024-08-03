{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  nixConfig = {
    bash-prompt-prefix = "(cuda-shell) ";
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
          };
        };
      in {
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            rust-bin.stable."1.80.0".default
            linuxPackages.nvidia_x11
            cudatoolkit
            python311
            python311Packages.torch-bin
            python311Packages.pillow
          ];

          shellHook = ''
            export CUDA_HOME=${pkgs.cudatoolkit}
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            echo "Checking PyTorch + CUDA..."
            python3 -c 'import torch; print("CUDA found!" if torch.cuda.is_available() else "CUDA not found.")'
          '';
        };
      }
    );
}