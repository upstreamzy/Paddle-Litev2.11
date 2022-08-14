#!/bin/bash
set -e

git submodule update --init --recursive

build_dir=build_fpga
mkdir -p ${build_dir}
cd ${build_dir}

GEN_CODE_PATH_PREFIX=lite/gen_code
mkdir -p ./${GEN_CODE_PATH_PREFIX}
touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=OFF \
        -DLITE_WITH_FPGA=ON \
        -DLITE_WITH_OPENMP=OFF \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
        -DWITH_TESTING=OFF \
        -DLITE_BUILD_EXTRA=OFF \
        -DLITE_WITH_PYTHON=OFF \
        -DLITE_WITH_PROFILE=OFF \
        -DLITE_WITH_LOG=OFF \
        -DLITE_WITH_CV=OFF  \
        -DLITE_ON_MODEL_OPTIMIZE_TOOL=OFF \
        #-DLITE_WITH_PYTHON=ON \
        #-DPY_VERSION="3.7"
        # -DWITH_STATIC_LIB=OFF \

# -DARM_TARGET_OS=armlinux \
# make clean
make -j8

cd -
