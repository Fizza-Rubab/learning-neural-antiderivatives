#include "common.h"
#include "texture.h"

__global__ void CopyTextureObjToTensor(const GenericTextureParams p)
{    
    // Calculate pixel position. // px and py are flipped to account for opengl's memory alloc
    int py = blockIdx.x * blockDim.x + threadIdx.x;
    int px = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    
    const cudaTextureObject_t texObj = p.cudaTexObj;
    
    // Check if the current thread position is within the texture dimensions.
    if (px < p.height && py < p.width && pz < p.channels){
        int pidx = (px + p.height * py) * p.channels + p.height * p.width * pz;
        int layerOffset = (p.height * p.width * p.channels) * (p.layerToTransfer);
        pidx += layerOffset;
        
        if(p.channels == 1){             
            float texValue = tex2D<float>(texObj, px, py);
            p.out[pidx] = texValue;
        }
        else if(p.channels == 2){
            float2 texValue = tex2D<float2>(texObj, px, py);
            p.out[pidx] = texValue.x;
            p.out[pidx+1] = texValue.y;
        }
        else if(p.channels == 3){
            float4 texValue = tex2D<float4>(texObj, px, py);
            p.out[pidx] = texValue.x;
            p.out[pidx+1] = texValue.y;
            p.out[pidx+2] = texValue.z;
            p.out[pidx+3] = 0.f;
        }
        else if(p.channels == 4){
            float4 texValue = tex2D<float4>(texObj, px, py);
            p.out[pidx] = texValue.x;
            p.out[pidx+1] = texValue.y;
            p.out[pidx+2] = texValue.z;
            p.out[pidx+3] = texValue.w;
        }
    }
}

__global__ void CopyTensorToSurfObj(const GenericTextureParams p)
{    
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    
    const cudaSurfaceObject_t surfObj = p.cudaSurfObj;
    const float* tensor = p.input;

    // Check if the current thread position is within the texture dimensions.
    if (px < p.width && py < p.height && pz < p.channels){
        int pidx = (px + p.width * py) * p.channels + p.height * p.width * pz;
        int layerOffset = (p.height * p.width * p.channels) * (p.layerToTransfer);
        pidx += layerOffset;

        if(p.channels == 1){
            float value = tensor[pidx];
            surf2Dwrite<float>(value, surfObj, px * sizeof(float), py, cudaBoundaryModeTrap);
        }
        else if(p.channels == 2){
            float2 value;
            value.x = tensor[pidx];
            value.y = tensor[pidx+1];
            surf2Dwrite<float2>(value, surfObj, px * 2 * sizeof(float), py, cudaBoundaryModeTrap);
        
        }
        else if(p.channels == 3){
            float4 value;
            value.x = tensor[pidx];
            value.y = tensor[pidx+1];
            value.z = tensor[pidx+2];
            value.w = 0;
            surf2Dwrite<float4>(value, surfObj, px * 4 * sizeof(float), py, cudaBoundaryModeTrap);
        
        }
        else if(p.channels == 4){
            float4 value;
            value.x = tensor[pidx];
            value.y = tensor[pidx+1];
            value.z = tensor[pidx+2];
            value.w = tensor[pidx+3];
            surf2Dwrite<float4>(value, surfObj, px * 4 * sizeof(float), py, cudaBoundaryModeTrap);
        }
    }

}
