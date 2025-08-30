#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <opencv2/core/cuda.hpp>
#include <mutex>
#include <atomic>

// === 在这里填写要捕获的窗口标题（忽略大小写） ===
static const wchar_t* kWindowTitle = L"Beat World (全网BGP线路）";

MIDL_INTERFACE("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")
IDirect3DDxgiInterfaceAccess : public ::IUnknown{
    virtual HRESULT STDMETHODCALLTYPE GetInterface(REFIID iid, void** p) = 0;
};

class WGCCaptureCuda {
public:
    WGCCaptureCuda() = default;
    ~WGCCaptureCuda();

    bool initFromWindowTitle(const wchar_t* title);
    bool init(HWND hwnd);
    void start();
    void stop();

    // 把最新帧拷到 GPU BGRA (CV_8UC4)，异步在给定 stream 上
    bool blitToGpuBgra(cv::cuda::GpuMat& outBgra, cudaStream_t stream);

    int width() const { return size_.Width; }
    int height() const { return size_.Height; }
    int frameCount() const { return (int)fc_.load(); }
    bool hasFrame() const { return gotFrame_; }

private:
    struct D3D11Pack {
        ID3D11Device* device = nullptr;
        ID3D11DeviceContext* context = nullptr;
        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrtDevice{ nullptr };
        ~D3D11Pack();
    };
    static D3D11Pack CreateD3D11();
    static HWND FindWindowByTitleInsensitive(const wchar_t* title);

    void recreateCopyTex(int w, int h);
    void destroyTex();
    void onArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender);

private:
    D3D11Pack d3d_;
    HWND hwnd_{};
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool_{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session_{ nullptr };
    winrt::event_token token_{};
    winrt::Windows::Graphics::SizeInt32 size_{};

    ID3D11Texture2D* copyTex_ = nullptr;
    cudaGraphicsResource* cudaRes_ = nullptr;

    std::mutex mtx_;
    std::atomic<int> fc_{ 0 };
    bool running_ = false;
    bool gotFrame_ = false;
};
