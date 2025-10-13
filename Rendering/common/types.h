#pragma once 

#include "common.h"
#include <vector>

// A simple class to store some info. 
class GraphicsResource{
    public:
        cudaGraphicsResource* cudaRes;
        cudaStream_t stream;
        cudaArray* cuda_texture_array;
        cudaTextureObject_t cudaTexObj;
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
};

class GenericTextureParams
{
    public:
        float*          input;                          // Incoming image buffer.
        int             width;                          // Image width.
        int             height;                         // Image height.
        int             channels;                       // No. of channels.
        int             layers;                         // No. of layers if layered texture
        float*          out;                            // Outgoing image attributes.
        size_t          imageSize;
        cudaTextureObject_t cudaTexObj;
        cudaSurfaceObject_t cudaSurfObj;
        int             layerToTransfer;
};
