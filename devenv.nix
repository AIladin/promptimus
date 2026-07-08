{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{

  languages.python = {

    version = "3.12";
    enable = true;
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  dotenv.enable = true;

  git-hooks.hooks = {
    ruff.enable = true;
  };

  delta.enable = true;
  # See full reference at https://devenv.sh/reference/options/
}
