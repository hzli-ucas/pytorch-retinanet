#include <torch/extension.h>

#ifdef WITH_CUDA
	#include "ATen/cuda/CUDAContext.h"
	#include "ATen/cuda/CUDAEvent.h"
    #include "THC.h"
    #include "cuda/nms_kernel.h"
    THCState* state = at::globalContext().lazyInitCUDA();
#endif

int cpu_nms(torch::Tensor keep_out, torch::Tensor num_out, torch::Tensor boxes, torch::Tensor order, torch::Tensor areas, float nms_overlap_thresh) {
  // boxes has to be sorted
  assert(keep_out.is_contiguous());
  assert(boxes.is_contiguous());
  assert(order.is_contiguous());
  assert(areas.is_contiguous());
  assert(keep_out.dtype() == torch::kInt64);
  assert(num_out.dtype() == torch::kInt64);
  assert(order.dtype() == torch::kInt64);
  // Number of ROIs
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  int64_t *keep_out_flat = (int64_t *)keep_out.data_ptr();
  float *boxes_flat = (float *)boxes.data_ptr();
  int64_t *order_flat = (int64_t *)order.data_ptr();
  float *areas_flat = (float *)areas.data_ptr();

  char *suppressed_flat =  new char[boxes_num];
  memset(suppressed_flat, 0, sizeof(char) * boxes_num);

  // nominal indices
  int i, j;
  // sorted indices
  int _i, _j;
  // temp variables for box i's (the box currently under consideration)
  float ix1, iy1, ix2, iy2, iarea;
  // variables for computing overlap with box j (lower scoring box)
  float xx1, yy1, xx2, yy2;
  float w, h;
  float inter, ovr;

  int64_t num_to_keep = 0;
  for (_i=0; _i < boxes_num; ++_i) {
    i = order_flat[_i];
    if (suppressed_flat[i] == 1) {
      continue;
    }
    keep_out_flat[num_to_keep++] = i;
    ix1 = boxes_flat[i * boxes_dim];
    iy1 = boxes_flat[i * boxes_dim + 1];
    ix2 = boxes_flat[i * boxes_dim + 2];
    iy2 = boxes_flat[i * boxes_dim + 3];
    iarea = areas_flat[i];
    for (_j = _i + 1; _j < boxes_num; ++_j) {
      j = order_flat[_j];
      if (suppressed_flat[j] == 1) {
        continue;
      }
      xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
      yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
      xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
      yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
      w = fmaxf(0.0, xx2 - xx1 + 1);
      h = fmaxf(0.0, yy2 - yy1 + 1);
      inter = w * h;
      ovr = inter / (iarea + areas_flat[j] - inter);
      if (ovr >= nms_overlap_thresh) {
        suppressed_flat[j] = 1;
      }
    }
  }

  delete[] suppressed_flat;
  int64_t *num_out_flat = (int64_t *)num_out.data_ptr();
  *num_out_flat = num_to_keep;
  return 1;
}

#ifdef WITH_CUDA
int gpu_nms(torch::Tensor keep_out, torch::Tensor num_out, torch::Tensor boxes, float nms_overlap_thresh) {
  // boxes has to be sorted
  assert(keep_out.is_contiguous());
  assert(boxes.is_contiguous());
  assert(keep_out.dtype() == torch::kInt64);
  assert(num_out.dtype() == torch::kInt64);

  _nms((long long *)keep_out.data_ptr(), (long long *)num_out.data_ptr(),
       boxes.size(0), (float *)boxes.data_ptr(), nms_overlap_thresh);

  return 1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_nms", &cpu_nms, "non-maximum supression function with cpu");
#ifdef WITH_CUDA
  m.def("gpu_nms", &gpu_nms, "non-maximum supression function with gpu");
#endif
}