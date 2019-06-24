// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>
#include <float.h>
#include "nms_kernel.h"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      printf("%s\n", cudaGetErrorString(error)); \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    unsigned long long t = 0;
    for (int i = start; i < col_size; i++) {
      const int box_col_idx = threadsPerBlock * col_start + i;
      if (devIoU(cur_box, dev_boxes + box_col_idx * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


void _nms(long long *keep_out, long long *num_out,
          const int boxes_num, const float *boxes_dev, const float nms_overlap_thresh) {

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  unsigned long long* mask_dev = NULL;
  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  unsigned long long *mask_host = new unsigned long long[boxes_num * col_blocks];
  CUDA_CHECK(cudaMemcpy(mask_host,
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  unsigned long long *remv_boxes = new unsigned long long[col_blocks];
  memset(remv_boxes, 0, sizeof(unsigned long long) * col_blocks);

  long long &num_to_keep = *num_out;
  num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    const int nblock = i / threadsPerBlock;
    const int inblock = i % threadsPerBlock;

    if (!(remv_boxes[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_boxes[j] |= p[j];
      }
    }
  }

  CUDA_CHECK(cudaFree(mask_dev));
  delete[] mask_host;
  delete[] remv_boxes;
}

#ifdef __cplusplus
}
#endif
