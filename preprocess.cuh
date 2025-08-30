#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// �� D3D11 ע��ӳ��õ��� cudaArray_t (BGRA8_UNORM) �� letterbox �� FP16 CHW (RGB, 0..1)
extern "C" void ppRunLetterboxFromCudaArray(
    cudaArray_t arr,           // Դ BGRA ����� cudaArray
    int srcW, int srcH,        // Դ��ͼ�ߴ磨�� WGC ֡�ߴ磩
    int roiX, int roiY,        // ROI ���Ͻǣ�����ͼ�����
    int roiW, int roiH,        // ROI ���
    __half* outCHW,            // Ŀ����� (FP16, CHW, RGB)
    int dstW, int dstH,        // Ŀ����������ߴ� (���� 640x640)
    float s,                   // letterbox ��������
    int pad_x, int pad_y,      // letterbox ����/���µ� padding
    int pad_114,               // 114���� 114 ���ɣ�
    cudaStream_t stream        // CUDA �������� TensorRT ����һ�£�
);
