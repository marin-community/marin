#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nv/target>

__global__ void shuttle_cuda_header_smoke(__half* fp16, __nv_bfloat16* bf16) {
  const int index = static_cast<int>(threadIdx.x);
  fp16[index] = __float2half(1.0f);
  bf16[index] = __float2bfloat16(1.0f);
}
