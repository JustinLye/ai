#!/bin/bash

sudo apt install -y \
  pipx \
  swig

pipx install poetry
pipx ensurepath
source ~/.bashrc
poetry install