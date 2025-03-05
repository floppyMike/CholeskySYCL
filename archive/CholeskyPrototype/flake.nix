{
  description = "CholeskyPrototype";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
      # Dev time (developing tools)
      devInputs = with pkgs; [
        valgrind
      ];
      # Build time (build tools; header libs)
      nativeBuildInputs = with pkgs; [
        cmake
        clang-tools
      ];
      # Run time (libs to link with)
      buildInputs = with pkgs; [
      ];

    in {
    # Utilized by `nix develop`
    devShell.x86_64-linux = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
      name = "prototype";
      inherit buildInputs;
      nativeBuildInputs = nativeBuildInputs ++ devInputs;
    };

    # Utilized by `nix build`
    defaultPackage.x86_64-linux = pkgs.clangStdenv.mkDerivation rec {
      pname = "prototype";
      version = "0.1.0";
      src = ./.;

      inherit nativeBuildInputs;
      inherit buildInputs;

      buildPhase = "make -j $NIX_BUILD_CORES";

      installPhase = ''
        runHook preInstall
        install -m755 -D prototype $out/bin/prototype
        runHook postInstall
      '';
    };

    # Utilized by `nix run`
    apps.x86_64-linux = {
      type = "app";
      program = self.packages.x86_64-linux.prototype;
    };
  };
}
