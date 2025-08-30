#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>


extern "C" void ppRunLetterboxFromCudaArray(
    cudaArray_t arr,           // 源 BGRA 纹理的 cudaArray
    int srcW, int srcH,        // 源整图尺寸（即 WGC 帧尺寸）
    int roiX, int roiY,        // ROI 左上角（在整图坐标里）
    int roiW, int roiH,        // ROI 宽高
    __half* outCHW,            // 目标输出 (FP16, CHW, RGB)
    int dstW, int dstH,        // 目标网络输入尺寸 (例如 640x640)
    float s,                   // letterbox 缩放因子
    int pad_x, int pad_y,      // letterbox 左右/上下的 padding
    int pad_114,               // 114
    cudaStream_t stream        // CUDA 流
);

