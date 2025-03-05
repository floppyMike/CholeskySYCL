#!/bin/bash
source ./load.sh
base=$(pwd)

build="PLSSVM/build"
rm -rf "$build"
mkdir -p "$build"

# rm -fr "envs/plssvm-build"
python -m venv "envs/plssvm-build"
source "envs/plssvm-build/bin/activate"
pip install wheel setuptools
pip install -r "PLSSVM/install/python_requirements.txt"

cmake -B "$build" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPLSSVM_ENABLE_PERFORMANCE_TRACKING=ON -DPLSSVM_TARGET_PLATFORMS="nvidia:sm_86" -DPLSSVM_ENABLE_LANGUAGE_BINDINGS=ON -DPLSSVM_ENABLE_TESTING=ON -DPLSSVM_GENERATE_TEST_FILE=ON PLSSVM
cmake --build "$build" -j

# cp -r "$build/_deps/fmt-src/include/fmt" "PLSSVM/include" # Install bug workaround
cmake --install "$build" --prefix "$build/install"

deactivate
