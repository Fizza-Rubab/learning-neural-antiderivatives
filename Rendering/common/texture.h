#pragma once
#include "types.h"

#define IP_FWD_MAX_KERNEL_BLOCK_WIDTH   32
#define IP_FWD_MAX_KERNEL_BLOCK_HEIGHT  32
// there seems to be a dependence on block size for the transfer 

// CUDA kernel
__global__ void CopyTextureObjToTensor(const GenericTextureParams p);
__global__ void CopyTensorToSurfObj(const GenericTextureParams p);