#include "WGCCaptureCuda.h"
#include <algorithm>
#include <cstdio>

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

static inline void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) { std::fprintf(stderr, "%s hr=0x%08X\n", msg, (unsigned)hr); std::exit(1); }
}
static inline void ThrowIfCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s CUDA: %s\n", msg, cudaGetErrorString(e)); std::exit(1); }
}

WGCCaptureCuda::D3D11Pack::~D3D11Pack() {
    winrtDevice = nullptr;
    if (context) context->Release();
    if (device) device->Release();
}
WGCCaptureCuda::D3D11Pack WGCCaptureCuda::CreateD3D11() {
    D3D11Pack r;
    D3D_FEATURE_LEVEL fls[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    };
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    ThrowIfFailed(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        flags, fls, ARRAYSIZE(fls), D3D11_SDK_VERSION,
        &r.device, nullptr, &r.context),
        "D3D11CreateDevice failed");

    winrt::com_ptr<IDXGIDevice> dxgi;
    r.device->QueryInterface(__uuidof(IDXGIDevice), dxgi.put_void());
    winrt::com_ptr<IInspectable> insp;
    CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), insp.put());
    r.winrtDevice = insp.as<IDirect3DDevice>();
    return r;
}


HWND WGCCaptureCuda::FindWindowByTitleInsensitive(const wchar_t* title) {
    if (!title || !*title) return nullptr;
    std::wstring target = title;
    std::transform(target.begin(), target.end(), target.begin(), ::towlower);

    struct Ctx { std::wstring target; HWND exact = nullptr, partial = nullptr; } ctx{ target };

    EnumWindows([](HWND h, LPARAM lp)->BOOL {
        if (!IsWindowVisible(h)) return TRUE;
        wchar_t buf[512]{};
        GetWindowTextW(h, buf, 512);
        if (!buf[0]) return TRUE;
        auto& c = *reinterpret_cast<Ctx*>(lp);
        std::wstring s = buf;
        std::transform(s.begin(), s.end(), s.begin(), ::towlower);
        if (!c.exact && s == c.target) { c.exact = h; return FALSE; }
        if (!c.partial && s.find(c.target) != std::wstring::npos) c.partial = h;
        return TRUE;
        }, (LPARAM)&ctx);
    return ctx.exact ? ctx.exact : ctx.partial;
}


WGCCaptureCuda::~WGCCaptureCuda() { stop(); destroyTex(); }

bool WGCCaptureCuda::initFromWindowTitle(const wchar_t* title) {
    HWND w = FindWindowByTitleInsensitive(title);
    if (!w) {
        std::fwprintf(stderr, L"[WGC] 找不到窗口: %s\n", title);
        return false;
    }
    return init(w);
}

bool WGCCaptureCuda::init(HWND hwnd) {
    hwnd_ = hwnd;
    winrt::init_apartment(winrt::apartment_type::single_threaded);

    if (!GraphicsCaptureSession::IsSupported()) {
        std::fprintf(stderr, "[WGC] Not supported on this OS\n");
        return false;
    }
    d3d_ = CreateD3D11();


    auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interop->CreateForWindow(hwnd_, winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
        reinterpret_cast<void**>(winrt::put_abi(item)));
    if (FAILED(hr) || !item) {
        std::fprintf(stderr, "[WGC] CreateForWindow failed\n");
        return false;
    }
    item_ = item;
    size_ = item_.Size();
    std::printf("[WGC] Capture size: %d x %d\n", size_.Width, size_.Height);


    framePool_ = Direct3D11CaptureFramePool::CreateFreeThreaded(
        d3d_.winrtDevice, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size_);

    token_ = framePool_.FrameArrived([this](auto&& sender, auto&&) {
        onArrived(sender);
        });
    session_ = framePool_.CreateCaptureSession(item_);

    recreateCopyTex(size_.Width, size_.Height);
    return true;
}

void WGCCaptureCuda::start() {
    if (session_) { session_.StartCapture(); running_ = true; }
}

void WGCCaptureCuda::stop() {
    if (running_) {
        if (framePool_) framePool_.FrameArrived(token_);
        session_ = nullptr;
        framePool_ = nullptr;
        item_ = nullptr;
        running_ = false;
    }
}

void WGCCaptureCuda::recreateCopyTex(int w, int h) {
    std::lock_guard<std::mutex> lk(mtx_);

    if (cudaRes_) {
        ThrowIfCuda(cudaGraphicsUnregisterResource(cudaRes_), "cudaGraphicsUnregisterResource");
        cudaRes_ = nullptr;
    }
    if (copyTex_) { copyTex_->Release(); copyTex_ = nullptr; }

    D3D11_TEXTURE2D_DESC d{};
    d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1;
    d.Format = DXGI_FORMAT_B8G8R8A8_UNORM; d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = 0; d.CPUAccessFlags = 0; d.MiscFlags = 0;
    ThrowIfFailed(d3d_.device->CreateTexture2D(&d, nullptr, &copyTex_), "CreateTexture2D copyTex_");


    ThrowIfCuda(cudaGraphicsD3D11RegisterResource(&cudaRes_, copyTex_, cudaGraphicsRegisterFlagsNone),
        "cudaGraphicsD3D11RegisterResource");
}

void WGCCaptureCuda::destroyTex() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (cudaRes_) {
        cudaGraphicsUnregisterResource(cudaRes_);
        cudaRes_ = nullptr;
    }
    if (copyTex_) {
        copyTex_->Release();
        copyTex_ = nullptr;
    }
}

void WGCCaptureCuda::onArrived(Direct3D11CaptureFramePool const& sender) {
    auto frame = sender.TryGetNextFrame();
    if (!frame) return;


    auto cs = frame.ContentSize();
    if (cs.Width != size_.Width || cs.Height != size_.Height) {
        size_ = cs;
        sender.Recreate(d3d_.winrtDevice, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size_);
        recreateCopyTex(size_.Width, size_.Height);
    }


    auto surf = frame.Surface();
    winrt::com_ptr<IDirect3DDxgiInterfaceAccess> access;
    try { access = surf.as<IDirect3DDxgiInterfaceAccess>(); }
    catch (...) { return; }

    winrt::com_ptr<ID3D11Texture2D> tex;
    if (FAILED(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), tex.put_void())))
        return;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (copyTex_) d3d_.context->CopyResource(copyTex_, tex.get());
        gotFrame_ = true;
    }
    fc_.fetch_add(1, std::memory_order_relaxed);
}

// 把最新帧拷到 GPU Mat（BGRA，CV_8UC4）
bool WGCCaptureCuda::blitToGpuBgra(cv::cuda::GpuMat& outBgra, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!copyTex_ || !cudaRes_ || !gotFrame_) return false;
    if (outBgra.empty() || outBgra.cols != size_.Width || outBgra.rows != size_.Height || outBgra.type() != CV_8UC4) {
        outBgra.create(size_.Height, size_.Width, CV_8UC4);
    }
    ThrowIfCuda(cudaGraphicsMapResources(1, &cudaRes_, stream), "cudaGraphicsMapResources");
    cudaArray* cuArray = nullptr;
    ThrowIfCuda(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaRes_, 0, 0),
        "cudaGraphicsSubResourceGetMappedArray");

    ThrowIfCuda(cudaMemcpy2DFromArrayAsync(
        outBgra.ptr<uchar>(), outBgra.step, cuArray, 0, 0,
        size_.Width * 4, size_.Height, cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DFromArrayAsync");
    ThrowIfCuda(cudaGraphicsUnmapResources(1, &cudaRes_, stream), "cudaGraphicsUnmapResources");
    return true;
}

