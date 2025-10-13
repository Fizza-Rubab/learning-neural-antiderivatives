/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 */
#pragma once

#include "torch_common.inl"
#include <tuple>
#include "../common/glutil.h"
#include "../common/types.h"


//------------------------------------------------------------------------
// Op prototypes. Return type macros for readability.
#define OP_RETURN_T     torch::Tensor
#define OP_RETURN_TT    std::tuple<torch::Tensor, torch::Tensor>
#define OP_RETURN_TTT   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTTT  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTV   std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define OP_RETURN_TTTTV std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define GResGTP         std::tuple<GraphicsResource, GenericTextureParams>

GResGTP                 RegisterTexture                 (GLuint tex, int layers, int width, int height, int channels, torch::Tensor out_tensor);
void                    TextureToTensor                 (GraphicsResource g, GenericTextureParams p);
void                    TensorToTexture                 (torch::Tensor img_tensor, GLuint tex);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Classes:
    pybind11::class_<GenericTextureParams>(m, "GenericTextureParams").def(pybind11::init<>());
    pybind11::class_<GraphicsResource>(m, "GraphicsResource").def(pybind11::init<>());

    // Functions:
    m.def("RegisterTexture",                  &RegisterTexture,                     "registering texture to cuda");
    m.def("TextureToTensor",                  &TextureToTensor,                     "mem copies texture to a tensor");
    m.def("TensorToTexture",                  &TensorToTexture,                     "mem copies tensor to a texture");
}
//------------------------------------------------------------------------



