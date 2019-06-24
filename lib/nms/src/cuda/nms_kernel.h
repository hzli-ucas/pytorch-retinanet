#ifndef _NMS_KERNEL
#define _NMS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void _nms(long long *keep_out, long long *num_out,
          const int boxes_num, const float *boxes_dev, const float nms_overlap_thresh);

#ifdef __cplusplus
}
#endif

#endif

