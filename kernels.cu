#include "kernels.h"
#include <cmath>

__global__ void interleavedToPlanarKernel(const float* src, size_t srcPitch, float* dst,
    int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    int y = idx / width;
    int x = idx % width;
    const float* row = (const float*)((const char*)src + y * srcPitch);
    float b = row[x * 3 + 0];
    float g = row[x * 3 + 1];
    float r = row[x * 3 + 2];
    dst[0 * total + idx] = b;
    dst[1 * total + idx] = g;
    dst[2 * total + idx] = r;
}

extern "C" void launchInterleavedToPlanarKernel(const float* src, size_t srcPitch, float* dst,
    int width, int height, cudaStream_t stream) {
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    interleavedToPlanarKernel << <blocks, threads, 0, stream >> > (src, srcPitch, dst, width, height);
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void drawBoxesKernel(uchar4* img, int W, int H,
    const float* detBoxes, const int* detClasses, const float* detScores,
    int N, float scale, int padX, int padY, int offX, int offY,
    float scoreThresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (detScores[i] < scoreThresh) return;

    // xyxy 是基于 letterbox 后的网络输入坐标
    float x1 = detBoxes[i * 4 + 0];
    float y1 = detBoxes[i * 4 + 1];
    float x2 = detBoxes[i * 4 + 2];
    float y2 = detBoxes[i * 4 + 3];

    // 逆 letterbox 回原窗口坐标，再加 ROI 偏移
    int xi1 = clampi((int)llroundf((x1 - padX) / scale) + offX, 0, W - 1);
    int yi1 = clampi((int)llroundf((y1 - padY) / scale) + offY, 0, H - 1);
    int xi2 = clampi((int)llroundf((x2 - padX) / scale) + offX, 0, W - 1);
    int yi2 = clampi((int)llroundf((y2 - padY) / scale) + offY, 0, H - 1);
    if (xi2 <= xi1 || yi2 <= yi1) return;

    uchar4 color = make_uchar4(0, 255, 0, 255);
    // 顶/底边
    for (int x = xi1; x <= xi2; ++x) {
        img[yi1 * W + x] = color;
        img[yi2 * W + x] = color;
    }
    // 左/右边
    for (int y = yi1; y <= yi2; ++y) {
        img[y * W + xi1] = color;
        img[y * W + xi2] = color;
    }
}

extern "C" void launchDrawBoxesKernel(uchar4* img, int imgWidth, int imgHeight,
    const float* detBoxes, const int* detClasses, const float* detScores,
    int numDet, float scale, int padX, int padY, int offsetX, int offsetY,
    float scoreThresh, cudaStream_t stream) {
    if (numDet <= 0) return;
    int threads = 64;
    int blocks = (numDet + threads - 1) / threads;
    drawBoxesKernel << <blocks, threads, 0, stream >> > (img, imgWidth, imgHeight,
        detBoxes, detClasses, detScores, numDet,
        scale, padX, padY, offsetX, offsetY, scoreThresh);
}

__global__ void drawVLinesKernel(uchar4* img, int W, int H, int x1, int x2, uchar4 c) {
    if (x1 >= 0 && x1 < W) {
        for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < H; y += blockDim.x * gridDim.x) {
            img[y * W + x1] = c;
        }
    }
    if (x2 >= 0 && x2 < W) {
        for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < H; y += blockDim.x * gridDim.x) {
            img[y * W + x2] = c;
        }
    }
}

extern "C" void launchDrawVLinesKernel(uchar4* img, int imgWidth, int imgHeight,
    int x1, int x2, uchar4 color, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((imgHeight + threads - 1) / threads, 1024);
    drawVLinesKernel << <blocks, threads, 0, stream >> > (img, imgWidth, imgHeight, x1, x2, color);
}

__global__ void drawRectKernel(uchar4* img, int W, int H, int x, int y, int w, int h, uchar4 c) {
    int x1 = clampi(x, 0, W - 1);
    int y1 = clampi(y, 0, H - 1);
    int x2 = clampi(x + w - 1, 0, W - 1);
    int y2 = clampi(y + h - 1, 0, H - 1);
    for (int xx = x1; xx <= x2; ++xx) {
        img[y1 * W + xx] = c;
        img[y2 * W + xx] = c;
    }
    for (int yy = y1; yy <= y2; ++yy) {
        img[yy * W + x1] = c;
        img[yy * W + x2] = c;
    }
}

extern "C" void launchDrawRectKernel(uchar4* img, int imgWidth, int imgHeight,
    int x, int y, int w, int h, uchar4 color, cudaStream_t stream) {
    drawRectKernel << <1, 1, 0, stream >> > (img, imgWidth, imgHeight, x, y, w, h, color);
}
