@echo off
set CUDA_ARCH=-gencode arch=compute_30,code=sm_30 ^
           -gencode arch=compute_35,code=sm_35 ^
           -gencode arch=compute_50,code=sm_50 ^
           -gencode arch=compute_52,code=sm_52 ^
           -gencode arch=compute_60,code=sm_60 ^
           -gencode arch=compute_61,code=sm_61
set VS_COMNTOOLS=%VS140COMNTOOLS%

:: Build NMS
cd nms/src/cuda
if not defined VSINSTALLDIR (
  echo Call "%VS_COMNTOOLS%vsvars32.bat"
  call "%VS_COMNTOOLS%vsvars32.bat"
)
echo Compiling nms kernels by nvcc...
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -MD %CUDA_ARCH%
cd ../../
echo Building cpp extension...
python build.py build
mkdir _ext >nul
cd build/
for /R %%f in (_ext\*) do (
  echo Move %%f to lib\nms\_ext\
  move %%f ..\_ext\ >nul
)
cd ../
echo Delete .\build
del/s/q .\build\*.* >nul
rd/s/q .\build >nul
cd ../
