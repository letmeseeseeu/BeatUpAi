#include <cuda_runtime.h>
#include <cuda_fp16.h>

static inline cudaTextureObject_t makeBGRA8Tex(cudaArray_t arr) {
    cudaResourceDesc r{};
    r.resType = cudaResourceTypeArray;
    r.res.array.array = arr;

    cudaTextureDesc t{};
    t.normalizedCoords = 0;                          // ÏñËØ×ø±ê
    t.filterMode = cudaFilterModeLinear;       // Ë«ÏßÐÔ
    t.addressMode[0] = cudaAddressModeClamp;
    t.addressMode[1] = cudaAddressModeClamp;
    t.readMode = cudaReadModeNormalizedFloat; // ¶Á³ö float4£¬·¶Î§ [0,1]

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &r, &t, nullptr);
    return tex;
}

__global__ void kLetterboxBGRA2RGB_FP16(
    cudaTextureObject_t tex, int srcW, int srcH,
    int roiX, int roiY, int roiW, int roiH,
    __half* __restrict__ outCHW, int dstW, int dstH,
    float s, int pad_x, int pad_y, float pad_f
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;

    // Ä¿±êÏñËØÓ³Éä»Ø ROI ÄÚ×ø±ê
    float sx = (dx - pad_x) / s;
    float sy = (dy - pad_y) / s;

    float R, G, B;
    if (sx >= 0.0f && sy >= 0.0f && sx <= roiW - 1.0f && sy <= roiH - 1.0f) {
        // °ëÏñËØ¶ÔÆë£»Ó³Éäµ½ÕûÍ¼×ø±ê£¨BGRA ÎÆÀí£©
        float u = (roiX + sx) + 0.5f;
        float v = (roiY + sy) + 0.5f;

        // ¶Á BGRA£¨x=B, y=G, z=R, w=A£©£¬·¶Î§ [0,1]
        float4 bgra = tex2D<float4>(tex, u, v);
        R = bgra.z;
        G = bgra.y;
        B = bgra.x;
    }
    else {
        R = G = B = pad_f; // 114/255
    }

    // Ð´ FP16 CHW
    int base = dy * dstW + dx;
    outCHW[0 * dstH * dstW + base] = __float2half(R);
    outCHW[1 * dstH * dstW + base] = __float2half(G);
    outCHW[2 * dstH * dstW + base] = __float2half(B);
}

extern "C" void ppRunLetterboxFromCudaArray(
    cudaArray_t arr, int srcW, int srcH,
    int roiX, int roiY, int roiW, int roiH,
    __half* outCHW, int dstW, int dstH,
    float s, int pad_x, int pad_y, int pad_114,
    cudaStream_t stream
) {
    cudaTextureObject_t tex = makeBGRA8Tex(arr);

    dim3 blk(16, 16);
    dim3 grd((dstW + blk.x - 1) / blk.x, (dstH + blk.y - 1) / blk.y);

    float pad_f = (float)pad_114 / 255.0f;

    kLetterboxBGRA2RGB_FP16 << <grd, blk, 0, stream >> > (
        tex, srcW, srcH, roiX, roiY, roiW, roiH,
        outCHW, dstW, dstH, s, pad_x, pad_y, pad_f
        );

    cudaDestroyTextureObject(tex);
}
