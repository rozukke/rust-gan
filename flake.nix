{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    systems.url = "github:nix-systems/x86_64-linux";
    pygan.url = "path:./pysrc";
    pygan.flake = false;
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  nixConfig = {
    bash-prompt-prefix = "(cuda-shell) ";
  };

  outputs = { self, pygan, nixpkgs, rust-overlay, flake-utils, ... }:
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
            nativeBuildInputs = with pkgs; [ makeWrapper ];
            buildInputs = with pkgs; [
              linuxPackages.nvidia_x11
              cudatoolkit
              python311
              python311Packages.torch-bin
              python311Packages.pillow
            ];

            # Makes location of Python sources available to the packaged Rust
            postInstall = ''
              for i in `find $out/bin -maxdepth 1 -type f -executable`; do
                wrapProgram $i --set PY_GAN_PATH ${pygan}
              done
            '';
          };

          # Default target for nix commands
          default = self.packages.${system}.rust-gan;

        };

        devShells.default = with pkgs; mkShell {
          buildInputs = [
            _rustToolchain
            linuxPackages.nvidia_x11
            cudatoolkit
            python311
            python311Packages.torch
            python311Packages.pillow
          ];

          env = {
            PY_GAN_PATH = "${pygan}";
          };

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
