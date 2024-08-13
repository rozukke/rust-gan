{
  description = "A dev environment for PyTorch with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    # Hack to make flake utils build for only one system
    systems.url = "github:nix-systems/x86_64-linux";

    # Hack to make local files available without rewriting the install step
    pygan.url = "path:./pysrc";
    pygan.flake = false;

    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  # Conda-style prefix to make being inside a shell less confusing
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
        
        # Make package build step use exact specified Rust version
        # Note that another hack is required to make the std library visible for code editors.
        _rustToolchain = pkgs.rust-bin.stable."1.80.0".default;
        _rustPlatform = pkgs.makeRustPlatform {
          rustc = _rustToolchain;
          cargo = _rustToolchain;
        };
        # Hack to make Python work since having the packages individually specified can break apparently
        _python311 = (pkgs.python311.withPackages (py: with py; [ torch-bin pillow ]));

        # Use Cargo manifest for package configuration
        manifest = (pkgs.lib.importTOML ./Cargo.toml).package;

      in {
        packages = {
          rust-gan = _rustPlatform.buildRustPackage {
            pname = manifest.name;
            inherit (manifest) version;

            src = pkgs.lib.cleanSource ./.;

            cargoLock.lockFile = ./Cargo.lock;
            nativeBuildInputs = with pkgs; [ makeWrapper ];
            # Not sure if propagatedBuildInputs is a requirement here but don't want to touch
            propagatedBuildInputs = with pkgs; [
              linuxPackages.nvidia_x11
              cudatoolkit
              _python311 
            ];
            
            # Make local file flake input available to program and Python available to PyO3
            env = {
              PYO3_PYTHON = "${_python311}/bin/python3";
              PY_GAN_PATH = "${pygan}";
            };

            preCheck = ''
              export RUST_BACKTRACE=1 
            '';

            # Makes location of Python sources available to the packaged Rust
            # Prepending the path is a hard requirement as otherwise the wrong Python interpreter is used.
            # CUDA will not work without manually setting LD Path either. Very fun.
            postInstall = ''
              for i in `find $out/bin -maxdepth 1 -type f -executable`; do
                wrapProgram $i --set PY_GAN_PATH ${pygan} \
                  --set PYO3_PYTHON ${_python311}/bin/python3 \
                  --prefix PATH : ${pkgs.lib.makeBinPath [ _python311 ]} \
                  --prefix LD_LIBRARY_PATH : /usr/lib/wsl/lib:${pkgs.lib.makeBinPath [ pkgs.linuxPackages.nvidia_x11 ]}
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
            _python311
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
