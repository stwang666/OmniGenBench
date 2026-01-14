#!/bin/bash
# Build script for Cython extension
# Usage: bash build_cython.sh

cd "$(dirname "$0")"
python setup.py build_ext --inplace

