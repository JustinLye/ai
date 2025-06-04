#!/bin/bash

sudo apt install \
  pipx \
  swig

pipx install poetry
pipx ensurepath
source ~/.bashrc
poetry install