{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    systems.url = "github:nix-systems/x86_64-linux";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  nixConfig = {
    bash-prompt-prefix = "(cuda-shell) ";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
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
        _rustToolchain = pkgs.rust-bin.stable."1.80.0".default;
        _rustPlatform = pkgs.makeRustPlatform {
          rustc = _rustToolchain;
          cargo = _rustToolchain;
        };
        manifest = (pkgs.lib.importTOML ./Cargo.toml).package;
      in {
        packages = {
          rust-gan = _rustPlatform.buildRustPackage {
            pname = manifest.name;
            inherit (manifest) version;

            src = pkgs.lib.cleanSource ./.;
            
            cargoLock.lockFile = ./Cargo.lock;
            buildInputs = with pkgs; [
              linuxPackages.nvidia_x11
              cudatoolkit
              python311
              python311Packages.torch-bin
              python311Packages.pillow
            ];

          };
          default = self.packages.${system}.rust-gan;
        };
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            _rustToolchain
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