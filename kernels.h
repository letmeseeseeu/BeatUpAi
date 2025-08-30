#pragma once
#include <cuda_runtime.h>

extern "C" {


    void launchInterleavedToPlanarKernel(const float* src, size_t srcPitch, float* dst,
        int width, int height, cudaStream_t stream);

    // 画检测框
    void launchDrawBoxesKernel(uchar4* img, int imgWidth, int imgHeight,
        const float* detBoxes, const int* detClasses, const float* detScores,
        int numDet, float scale, int padX, int padY, int offsetX, int offsetY,
        float scoreThresh, cudaStream_t stream);

    // 画两条竖直阈值线
    void launchDrawVLinesKernel(uchar4* img, int imgWidth, int imgHeight,
        int x1, int x2, uchar4 color, cudaStream_t stream);

    // 画屏蔽区矩形
    void launchDrawRectKernel(uchar4* img, int imgWidth, int imgHeight,
        int x, int y, int w, int h, uchar4 color, cudaStream_t stream);

}

