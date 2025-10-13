#pragma once 

#include <cuda_runtime.h>
#include "torch_common.inl"
#include "../common/glutil.h"
#include "../common/texture.h"
#include <iostream>
#include <vector>

// Init function - returns a tuple of custom structs
std::tuple<GraphicsResource, GenericTextureParams> RegisterTexture(GLuint tex, int layers, int width, int height, int channels, torch::Tensor out_tensor){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Sanity checks.
    ISML_CHECK(height > 0 && width > 0 && channels > 0, "Input texture must have shape[>0, >0, >0, *]");
    
    GenericTextureParams p = {};
    p.height       = height;
    p.width        = width;
    p.channels     = channels;
    p.layers       = layers;
    p.out = out_tensor.data_ptr<float>();

    cudaGraphicsResource* CudaRes = nullptr;

    // We always use GL_TEXTURE_2D_ARRAY here for Registering.
    // GL_TEXTURE_2D could be thought of as GL_TEXTURE_2D_ARRAY with just 1 layer.
    // It doesnt affect what we return at the end

    ISML_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&CudaRes, tex, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsNone)); // Doesnt support RGB format (workaround implemented)
    cudaArray *cuda_texture_array;
    
    cudaTextureObject_t cudaTexObj;
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    GraphicsResource g;
    g.cudaRes = CudaRes;
    g.stream = stream;
    g.cuda_texture_array = cuda_texture_array;
    g.cudaTexObj = cudaTexObj;
    g.resDesc = resDesc;
    g.texDesc = texDesc;

    return std::make_tuple(g, p);
}

//-------------------------------------------------------------------------
void TextureToTensor(GraphicsResource g, GenericTextureParams p){
    // Get info stored in the structs 
    auto CudaRes = g.cudaRes;
    auto stream = g.stream;
    auto cuda_texture_array = g.cuda_texture_array;
    auto cudaTexObj = g.cudaTexObj;
    auto resDesc = g.resDesc;
    auto texDesc = g.texDesc;

    // Map Graphics resource(opengl) to cuda and assign the correct pointer to the texture
    ISML_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &CudaRes, stream));

    for (int layer=0; layer<p.layers; layer ++){
        ISML_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_array, CudaRes, layer, 0));
        p.layerToTransfer = layer;
        // link cuda texture array to the texture object
        resDesc.res.array.array = cuda_texture_array;
        cudaCreateTextureObject(&cudaTexObj, &resDesc, &texDesc, NULL);

        dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, 1);

        p.cudaTexObj = cudaTexObj;
        // Launch CUDA kernel.
        void* args[] = {&p};
        void* func = (void*) CopyTextureObjToTensor;
        ISML_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

        ISML_CHECK_CUDA_ERROR(cudaDestroyTextureObject(cudaTexObj););
        
    }
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    ISML_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &CudaRes););
}


// -------------------------------------------------------------------------

// Tranfer contents of the input tensor to the texture (GPU -> GPU)
void TensorToTexture(torch::Tensor img_tensor, GLuint tex){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Check inputs.
    ISML_CHECK_DEVICE(img_tensor);
    ISML_CHECK_CONTIGUOUS(img_tensor);
    ISML_CHECK_F32(img_tensor);

    cudaGraphicsResource* CudaRes;
    cudaArray *cuda_texture_array;
    GenericTextureParams p;

    p.channels = img_tensor.size(3);
    p.width = img_tensor.size(2);
    p.height = img_tensor.size(1);
    p.layers = img_tensor.size(0);
    p.input = img_tensor.data_ptr<float>(); // pointer to the tensor
    size_t layersize = p.width * p.height * p.channels;

    ISML_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&CudaRes, tex, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsNone));
    // Map Graphics resource(opengl) to cuda and assign the correct pointer to the texture
    ISML_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &CudaRes, stream));
    
    // Surface object attributes
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray; 

    cudaSurfaceObject_t out_surface;
    /////////////////////////

    for (int layer = 0; layer < p.layers; layer++){
        p.layerToTransfer = layer;
        size_t layer_offset = layer * layersize;
        
        ISML_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_array, CudaRes, layer, 0)); // the third parameter controls the layers in array textures; the 4th controls mipmap level
        
        res_desc.res.array.array = cuda_texture_array;
        cudaCreateSurfaceObject(&out_surface, &res_desc);
        p.cudaSurfObj = out_surface;

        // Launch CUDA kernel.
        dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, 1);
        void* args[] = {&p};
        void* func = (void*) CopyTensorToSurfObj;
        ISML_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
    }
    // Unmap to access texture in openGL
    ISML_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &CudaRes, stream));
    ISML_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(CudaRes));

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
 
}