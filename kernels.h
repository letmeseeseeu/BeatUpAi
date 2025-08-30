#pragma once
#include <cuda_runtime.h>

extern "C" {

    // HWC (CV_32FC3) �� CHW float
    void launchInterleavedToPlanarKernel(const float* src, size_t srcPitch, float* dst,
        int width, int height, cudaStream_t stream);

    // ����������Ϊ letterbox �� xyxy���� scale/pad ����ӳ�䣻�ɶ���� ROI ƫ�ƣ�
    void launchDrawBoxesKernel(uchar4* img, int imgWidth, int imgHeight,
        const float* detBoxes, const int* detClasses, const float* detScores,
        int numDet, float scale, int padX, int padY, int offsetX, int offsetY,
        float scoreThresh, cudaStream_t stream);

    // ��������ֱ��ֵ��
    void launchDrawVLinesKernel(uchar4* img, int imgWidth, int imgHeight,
        int x1, int x2, uchar4 color, cudaStream_t stream);

    // �����������Σ�1���رߣ�
    void launchDrawRectKernel(uchar4* img, int imgWidth, int imgHeight,
        int x, int y, int w, int h, uchar4 color, cudaStream_t stream);

}
