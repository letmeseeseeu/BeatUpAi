// ======================= WGC + TensorRT10 + DD (No-Display, CUDA–D3D11 Interop, EfficientNMS, FP16-Input, Per-Key Thresholds) =======================
#define _WIN32_WINNT 0x0A00
#define WINVER        0x0A00
#define _CRT_SECURE_NO_WARNINGS

// -------- Windows / DirectX / WinRT --------
#include <windows.h>
#include <inspectable.h>
#include <shellscalingapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "shcore.lib")
#pragma comment(lib, "windowsapp.lib")

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

// -------- CUDA / TensorRT 10 --------
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cudaD3D11.h>

#pragma comment(lib, "cuda.lib")

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>

// -------- OpenCV（只用到 Rect2f）--------
#if __has_include(<opencv2/core.hpp>)
#   include <opencv2/core.hpp>
#elif __has_include(<opencv4/opencv2/core.hpp>)
#   include <opencv4/opencv2/core.hpp>
#else
#   error "OpenCV core headers not found"
#endif

// -------- STL --------
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cfloat>
#include <iomanip>
#include <limits>
#include <array>

// 提升计时精度 & Sleep 精度
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "DD.h"
#include "preprocess.cuh"

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;

// 声明 IDirect3DDxgiInterfaceAccess
MIDL_INTERFACE("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")
IDirect3DDxgiInterfaceAccess : public ::IUnknown{
    virtual HRESULT STDMETHODCALLTYPE GetInterface(REFIID iid, void** p) = 0;
};

// ======================= 小工具 =======================
static std::filesystem::path getExeDir() {
    wchar_t buf[MAX_PATH]{}; DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
}
static std::filesystem::path toExeDir(const std::string& p) {
    std::filesystem::path pp = std::filesystem::u8path(p);
    if (pp.is_absolute()) return pp; return getExeDir() / pp;
}
template<typename T> static inline T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline double nowSec() {
    using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
static inline void toUpperInplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return (char)std::toupper(c); });
}

// ======================= DD =======================
static const wchar_t* kDD_DLL_PATH = L"D:\\PythonCode\\BeatUp_YOLO_V8\\DD\\dd.54900\\dd.54900.dll";
static CDD   g_dd;
static bool  g_dd_ok = false;
static bool InitDD() {
    int ret = g_dd.GetFunAddr(kDD_DLL_PATH);
    switch (ret) {
    case 1: {
        int st = g_dd.DD_btn(0);
        if (st == 1) { std::wcout << L"[DD] Load OK: " << kDD_DLL_PATH << L"\n"; g_dd_ok = true; return true; }
        std::wcerr << L"[DD] Initialize Error: DD_btn(0) -> " << st << L"\n"; g_dd_ok = false; return false;
    }
    case -1: case -2: case -3:
        std::wcerr << L"[DD] Load Error: GetFunAddr -> " << ret << L"\n"; g_dd_ok = false; return false;
    default:
        std::wcerr << L"[DD] Error: GetFunAddr -> " << ret << L"\n"; g_dd_ok = false; return false;
    }
}
// 兼容保留（不再使用）
static inline void DDPress(int ddcode, int down_ms = 10) {
    if (!g_dd_ok) return; g_dd.DD_key(ddcode, 1); if (down_ms > 0) ::Sleep(down_ms); g_dd.DD_key(ddcode, 2);
}

// ======================= 异步发键器（SPSC 环形队列） =======================
class KeyTapSender {
public:
    struct Tap { int code; int down_ms; };

    void start() {
        if (started_) return;
        started_ = true;
        timeBeginPeriod(1);
        run_.store(true, std::memory_order_release);
        worker_ = std::thread([this] { this->loop(); });
        SetThreadPriority(worker_.native_handle(), THREAD_PRIORITY_HIGHEST);
    }
    void stop() {
        if (!started_) return;
        run_.store(false, std::memory_order_release);
        if (worker_.joinable()) worker_.join();
        timeEndPeriod(1);
        started_ = false;
    }
    ~KeyTapSender() { stop(); }

    inline void enqueue(int code, int down_ms) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) {
            // 满：丢弃最旧，避免堆积
            tail_.store((tail_.load(std::memory_order_relaxed) + 1) & MASK, std::memory_order_release);
        }
        buf_[head] = Tap{ code, down_ms };
        head_.store(next, std::memory_order_release);
    }
    int depth() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return (int)((h >= t) ? (h - t) : (CAP - (t - h)));
    }

private:
    void loop() {
        while (run_.load(std::memory_order_acquire)) {
            size_t t = tail_.load(std::memory_order_relaxed);
            if (t == head_.load(std::memory_order_acquire)) {
                Sleep(0);
                continue;
            }
            Tap ev = buf_[t];
            tail_.store((t + 1) & MASK, std::memory_order_release);

            if (g_dd_ok) {
                g_dd.DD_key(ev.code, 1);
                if (ev.down_ms > 0) {
                    if (ev.down_ms >= 2) Sleep(ev.down_ms - 1);
                    LARGE_INTEGER f, s, e; QueryPerformanceFrequency(&f);
                    QueryPerformanceCounter(&s);
                    const double target = (ev.down_ms / 1000.0);
                    for (;;) {
                        QueryPerformanceCounter(&e);
                        double dt = double(e.QuadPart - s.QuadPart) / double(f.QuadPart);
                        if (dt >= target) break;
                        YieldProcessor();
                    }
                }
                g_dd.DD_key(ev.code, 2);
            }
        }
    }

private:
    static constexpr size_t CAP = 256;
    static constexpr size_t MASK = CAP - 1;
    static_assert((CAP& (CAP - 1)) == 0, "CAP must be power of two");

    std::array<Tap, CAP> buf_{};
    std::atomic<size_t> head_{ 0 }, tail_{ 0 };
    std::atomic<bool> run_{ false };
    std::thread worker_{};
    bool started_{ false };
};

// ======================= D3D11 设备 =======================
struct D3D11Pack {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrtDevice{ nullptr };
    ~D3D11Pack() { winrtDevice = nullptr; if (context) context->Release(); if (device) device->Release(); }
};
static D3D11Pack CreateD3D11() {
    D3D11Pack r;
    D3D_FEATURE_LEVEL fls[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, fls, ARRAYSIZE(fls),
        D3D11_SDK_VERSION, &r.device, nullptr, &r.context);
    if (FAILED(hr)) { std::fprintf(stderr, "D3D11CreateDevice failed 0x%08X\n", (unsigned)hr); std::exit(1); }
    winrt::com_ptr<IDXGIDevice> dxgi; r.device->QueryInterface(__uuidof(IDXGIDevice), dxgi.put_void());
    winrt::com_ptr<IInspectable> insp;
    CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), insp.put());
    r.winrtDevice = insp.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    return r;
}

// ======================= WGC 捕获（CUDA–D3D11 互操作） =======================
class WGCCapture {
public:
    explicit WGCCapture(HWND hwnd) : hwnd_(hwnd) {}
    ~WGCCapture() { stop(); destroyUnsafe(); }

    bool init() {
        if (!GraphicsCaptureSession::IsSupported()) { std::fprintf(stderr, "[WGC] Not supported\n"); return false; }
        d3d_ = CreateD3D11();

        // CUDA 与 D3D11 绑定同卡
        {
            winrt::com_ptr<IDXGIDevice> dxgiDevice;
            HRESULT hrQI = d3d_.device->QueryInterface(__uuidof(IDXGIDevice), dxgiDevice.put_void());
            if (FAILED(hrQI) || !dxgiDevice) { std::fprintf(stderr, "[WGC] QI IDXGIDevice failed\n"); return false; }
            winrt::com_ptr<IDXGIAdapter> adapter;
            HRESULT hrGA = dxgiDevice->GetAdapter(adapter.put());
            if (FAILED(hrGA) || !adapter) { std::fprintf(stderr, "[WGC] GetAdapter failed\n"); return false; }

            CUdevice cuDev{};
            CUresult r = cuD3D11GetDevice(&cuDev, adapter.get());
            if (r != CUDA_SUCCESS) { std::fprintf(stderr, "[WGC] cuD3D11GetDevice failed (%d)\n", (int)r); return false; }

            char busId[64] = { 0 };
            cuDeviceGetPCIBusId(busId, (int)sizeof(busId), cuDev);
            int ordinal = 0;
            cudaError_t ce = cudaDeviceGetByPCIBusId(&ordinal, busId);
            if (ce != cudaSuccess) { std::fprintf(stderr, "[WGC] cudaDeviceGetByPCIBusId failed: %s\n", cudaGetErrorString(ce)); return false; }
            cudaSetDevice(ordinal);
        }

        auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        GraphicsCaptureItem item{ nullptr };
        HRESULT hr = interop->CreateForWindow(hwnd_, winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
            reinterpret_cast<void**>(winrt::put_abi(item)));
        if (FAILED(hr) || !item) { std::fprintf(stderr, "[WGC] CreateForWindow failed\n"); return false; }
        item_ = item;

        size_ = item_.Size();
        w_.store(size_.Width, std::memory_order_relaxed);
        h_.store(size_.Height, std::memory_order_relaxed);
        std::printf("[WGC] Capture size: %d x %d\n", size_.Width, size_.Height);

        framePool_ = Direct3D11CaptureFramePool::CreateFreeThreaded(
            d3d_.winrtDevice, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size_);

        token_ = framePool_.FrameArrived([this](auto&& sender, auto&&) { onArrived(sender); });
        session_ = framePool_.CreateCaptureSession(item_);
        recreateGPU(size_.Width, size_.Height);
        return true;
    }

    void start() { session_.StartCapture(); running_.store(true, std::memory_order_release); }
    void stop() {
        if (running_.load(std::memory_order_acquire)) {
            if (framePool_) framePool_.FrameArrived(token_);
            session_ = nullptr; framePool_ = nullptr; item_ = nullptr;
            running_.store(false, std::memory_order_release);
        }
    }

    bool acquireLatestCudaArray(cudaArray_t& outArr, int& outW, int& outH, int& outBufIdx) {
        int idx = latest_.load(std::memory_order_acquire);
        if (idx < 0) return false;
        if (!gpuTex_[idx] || !cuRes_[idx]) return false;
        CUresult r1 = cuGraphicsMapResources(1, &cuRes_[idx], 0);
        if (r1 != CUDA_SUCCESS) return false;
        CUarray cuArr{};
        CUresult r2 = cuGraphicsSubResourceGetMappedArray(&cuArr, cuRes_[idx], 0, 0);
        if (r2 != CUDA_SUCCESS) { cuGraphicsUnmapResources(1, &cuRes_[idx], 0); return false; }

        outArr = reinterpret_cast<cudaArray_t>(cuArr);
        outW = w_.load(std::memory_order_acquire);
        outH = h_.load(std::memory_order_acquire);
        outBufIdx = idx;
        return true;
    }
    void releaseCudaArray(int bufIdx) {
        if (bufIdx < 0) return;
        cuGraphicsUnmapResources(1, &cuRes_[bufIdx], 0);
    }

    int width()  const { return w_.load(std::memory_order_acquire); }
    int height() const { return h_.load(std::memory_order_acquire); }

private:
    enum { kBufCount = 3 };

    void destroyUnsafe() {
        for (int i = 0; i < kBufCount; ++i) {
            if (cuRes_[i]) { cuGraphicsUnregisterResource(cuRes_[i]); cuRes_[i] = nullptr; }
            if (gpuTex_[i]) { gpuTex_[i]->Release(); gpuTex_[i] = nullptr; }
        }
        latest_.store(-1, std::memory_order_release);
        writeCursor_ = 0;
    }

    void recreateGPU(int w, int h) {
        std::lock_guard<std::mutex> lk(resMtx_);
        for (int i = 0; i < kBufCount; ++i) {
            if (cuRes_[i]) { cuGraphicsUnregisterResource(cuRes_[i]); cuRes_[i] = nullptr; }
            if (gpuTex_[i]) { gpuTex_[i]->Release(); gpuTex_[i] = nullptr; }
        }
        D3D11_TEXTURE2D_DESC td{};
        td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.MiscFlags = 0;

        for (int i = 0; i < kBufCount; ++i) {
            HRESULT hr = d3d_.device->CreateTexture2D(&td, nullptr, &gpuTex_[i]);
            if (FAILED(hr) || !gpuTex_[i]) { std::fprintf(stderr, "[WGC] CreateTexture2D failed\n"); std::exit(2); }
            CUresult r = cuGraphicsD3D11RegisterResource(&cuRes_[i], gpuTex_[i], 0);
            if (r != CUDA_SUCCESS) { std::fprintf(stderr, "[WGC] cuGraphicsD3D11RegisterResource failed\n"); std::exit(2); }
        }
        latest_.store(-1, std::memory_order_release);
        writeCursor_ = 0;
        w_.store(w, std::memory_order_release);
        h_.store(h, std::memory_order_release);
    }

    void onArrived(Direct3D11CaptureFramePool const& sender) {
        auto frame = sender.TryGetNextFrame();
        if (!frame) return;
        auto cs = frame.ContentSize();
        if (cs.Width != size_.Width || cs.Height != size_.Height) {
            size_ = cs;
            sender.Recreate(d3d_.winrtDevice, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size_);
            recreateGPU(size_.Width, size_.Height);
        }
        auto surf = frame.Surface();
        winrt::com_ptr<IDirect3DDxgiInterfaceAccess> access;
        try { access = surf.as<IDirect3DDxgiInterfaceAccess>(); }
        catch (...) { return; }
        winrt::com_ptr<ID3D11Texture2D> src;
        if (FAILED(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), src.put_void()))) return;

        int myIndex = writeCursor_;
        writeCursor_ = (writeCursor_ + 1) % kBufCount;

        {
            std::lock_guard<std::mutex> lk(resMtx_);
            if (!gpuTex_[myIndex]) return;
            d3d_.context->CopyResource(gpuTex_[myIndex], src.get());
        }

        latest_.store(myIndex, std::memory_order_release);
        fc_.fetch_add(1, std::memory_order_relaxed);
    }

private:
    struct D3D11Pack {
        ID3D11Device* device = nullptr;
        ID3D11DeviceContext* context = nullptr;
        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrtDevice{ nullptr };
        ~D3D11Pack() { winrtDevice = nullptr; if (context) context->Release(); if (device) device->Release(); }
    } d3d_;
    HWND hwnd_{};
    GraphicsCaptureItem item_{ nullptr };
    Direct3D11CaptureFramePool framePool_{ nullptr };
    GraphicsCaptureSession session_{ nullptr };
    winrt::event_token token_{};
    winrt::Windows::Graphics::SizeInt32 size_{};

    std::mutex resMtx_;
    ID3D11Texture2D* gpuTex_[kBufCount]{};
    CUgraphicsResource cuRes_[kBufCount]{};

    std::atomic<int> latest_{ -1 };
    int writeCursor_ = 0;

    std::atomic<int> fc_{ 0 };
    std::atomic<bool> running_{ false };

    std::atomic<int> w_{ 0 }, h_{ 0 };

public:
    ID3D11Device* device() const { return d3d_.device; }
    ID3D11DeviceContext* context() const { return d3d_.context; }

private:
    static D3D11Pack CreateD3D11() {
        D3D11Pack r;
        D3D_FEATURE_LEVEL fls[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, fls, ARRAYSIZE(fls),
            D3D11_SDK_VERSION, &r.device, nullptr, &r.context);
        if (FAILED(hr)) { std::fprintf(stderr, "D3D11CreateDevice failed 0x%08X\n", (unsigned)hr); std::exit(1); }
        winrt::com_ptr<IDXGIDevice> dxgi; r.device->QueryInterface(__uuidof(IDXGIDevice), dxgi.put_void());
        winrt::com_ptr<IInspectable> insp;
        CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), insp.put());
        r.winrtDevice = insp.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
        return r;
    }
};

// ======================= YOLO + TensorRT =======================
using namespace nvinfer1;
#define CHECK_CUDA(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

static inline float iouRect(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area(); float uni = a.area() + b.area() - inter + 1e-6f; return inter / uni;
}
struct Det { int cls; float conf; cv::Rect2f box; };

class TRTLogger : public ILogger {
    void log(Severity s, const char* msg) noexcept override { if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n"; }
} gLogger;

class YoloTRT {
public:
    bool init(const std::string& engineFile) {
        auto path = toExeDir(engineFile);
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Engine not found: " << path.string() << "\n"; return false; }
        f.seekg(0, std::ios::end); size_t sz = (size_t)f.tellg(); f.seekg(0, std::ios::beg);
        std::vector<char> buf(sz); f.read(buf.data(), sz);

        initLibNvInferPlugins(&gLogger, "");
        runtime_.reset(createInferRuntime(gLogger));
        if (!runtime_) { std::cerr << "createInferRuntime failed\n"; return false; }
        engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()));
        if (!engine_) { std::cerr << "deserializeCudaEngine failed\n"; return false; }
        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) { std::cerr << "createExecutionContext failed\n"; return false; }

        // 输入名 & 形状
        int nIO = engine_->getNbIOTensors();
        for (int i = 0; i < nIO; ++i) {
            const char* nm = engine_->getIOTensorName(i);
            if (!nm) continue;
            if (engine_->getTensorIOMode(nm) == TensorIOMode::kINPUT) inName_ = nm;
        }
        if (inName_.empty()) { std::cerr << "input tensor not found\n"; return false; }

        // ★ 确认输入是 FP16
        nvinfer1::DataType inDT = engine_->getTensorDataType(inName_.c_str());
        if (inDT != nvinfer1::DataType::kHALF) {
            std::cerr << "[TRT] Engine input is not FP16. Found: " << (int)inDT << "\n";
            return false;
        }

        Dims in = engine_->getTensorShape(inName_.c_str());
        N_ = in.d[0] > 0 ? in.d[0] : 1;
        C_ = in.d[1] > 0 ? in.d[1] : 3;
        H_ = in.d[2] > 0 ? in.d[2] : 640;
        W_ = in.d[3] > 0 ? in.d[3] : 640;
        if (in.d[0] < 0 || in.d[2] < 0 || in.d[3] < 0) {
            Dims4 fix{ 1,3,H_,W_ };
            if (!ctx_->setInputShape(inName_.c_str(), fix)) {
                std::cerr << "setInputShape failed\n"; return false;
            }
        }

        // NMS 输出名
        for (int i = 0; i < nIO; ++i) {
            const char* nm = engine_->getIOTensorName(i);
            if (!nm) continue;
            if (engine_->getTensorIOMode(nm) == TensorIOMode::kOUTPUT) {
                std::string s = nm;
                if (s.find("num_dets") != std::string::npos) outNum_ = s;
                else if (s.find("boxes") != std::string::npos) outBoxes_ = s;
                else if (s.find("scores") != std::string::npos) outScores_ = s;
                else if (s.find("classes") != std::string::npos) outClasses_ = s;
            }
        }
        hasNMS_ = (!outNum_.empty() && !outBoxes_.empty() && !outScores_.empty() && !outClasses_.empty());

        // 分配输入/输出
        CHECK_CUDA(cudaMalloc(&dIn_, sizeof(__half) * (size_t)N_ * C_ * H_ * W_));
        CHECK_CUDA(cudaStreamCreate(&stream_));

        if (hasNMS_) {
            auto shapeB = ctx_->getTensorShape(outBoxes_.c_str()); // [N, max, 4]
            maxDet_ = (shapeB.nbDims >= 3 ? shapeB.d[1] : 100);
            if (maxDet_ <= 0) maxDet_ = 100;
            CHECK_CUDA(cudaMalloc(&dNum_, sizeof(int) * N_));
            CHECK_CUDA(cudaMalloc(&dBoxes_, sizeof(float) * N_ * maxDet_ * 4));
            CHECK_CUDA(cudaMalloc(&dScores_, sizeof(float) * N_ * maxDet_));
            CHECK_CUDA(cudaMalloc(&dCls_, sizeof(int) * N_ * maxDet_));
            CHECK_CUDA(cudaMallocHost((void**)&hNum_, sizeof(int) * N_));
            CHECK_CUDA(cudaMallocHost((void**)&hBoxes_, sizeof(float) * N_ * maxDet_ * 4));
            CHECK_CUDA(cudaMallocHost((void**)&hScores_, sizeof(float) * N_ * maxDet_));
            CHECK_CUDA(cudaMallocHost((void**)&hCls_, sizeof(int) * N_ * maxDet_));
        }
        else {
            std::cerr << "[TRT] Engine missing EfficientNMS outputs.\n";
            return false;
        }
        return true;
    }

    void destroy() {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (dIn_) { cudaFree(dIn_); dIn_ = nullptr; }
        if (hasNMS_) {
            if (dNum_)    cudaFree(dNum_);
            if (dBoxes_)  cudaFree(dBoxes_);
            if (dScores_) cudaFree(dScores_);
            if (dCls_)    cudaFree(dCls_);
            if (hNum_)    cudaFreeHost(hNum_);
            if (hBoxes_)  cudaFreeHost(hBoxes_);
            if (hScores_) cudaFreeHost(hScores_);
            if (hCls_)    cudaFreeHost(hCls_);
        }
        ctx_.reset(); engine_.reset(); runtime_.reset();
    }

    bool enqueueAndCollect(float conf_thr,
        std::vector<Det>& out,
        float s, int pad_x, int pad_y,
        int roiW, int roiH)
    {
        out.clear();
        if (!ctx_->setTensorAddress(inName_.c_str(), dIn_)) { std::cerr << "setTensor(in) fail\n"; return false; }
        ctx_->setTensorAddress(outNum_.c_str(), dNum_);
        ctx_->setTensorAddress(outBoxes_.c_str(), dBoxes_);
        ctx_->setTensorAddress(outScores_.c_str(), dScores_);
        ctx_->setTensorAddress(outClasses_.c_str(), dCls_);

        if (!ctx_->enqueueV3(stream_)) { std::cerr << "enqueueV3 fail\n"; return false; }

        CHECK_CUDA(cudaMemcpyAsync(hNum_, dNum_, sizeof(int) * N_, cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaMemcpyAsync(hBoxes_, dBoxes_, sizeof(float) * N_ * maxDet_ * 4, cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaMemcpyAsync(hScores_, dScores_, sizeof(float) * N_ * maxDet_, cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaMemcpyAsync(hCls_, dCls_, sizeof(int) * N_ * maxDet_, cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        int num = hNum_[0];
        num = std::max(0, std::min(num, maxDet_));
        out.reserve(num);
        for (int i = 0; i < num; ++i) {
            float x1_ = hBoxes_[i * 4 + 0];
            float y1_ = hBoxes_[i * 4 + 1];
            float x2_ = hBoxes_[i * 4 + 2];
            float y2_ = hBoxes_[i * 4 + 3];
            float sc = hScores_[i];
            int   cls = hCls_[i];
            if (sc < conf_thr) continue;

            float x1 = (x1_ - pad_x) / s;
            float y1 = (y1_ - pad_y) / s;
            float x2 = (x2_ - pad_x) / s;
            float y2 = (y2_ - pad_y) / s;
            x1 = std::clamp(x1, 0.f, (float)roiW - 1); y1 = std::clamp(y1, 0.f, (float)roiH - 1);
            x2 = std::clamp(x2, 0.f, (float)roiW - 1); y2 = std::clamp(y2, 0.f, (float)roiH - 1);
            if (x2 <= x1 || y2 <= y1) continue;

            out.push_back({ cls, sc, cv::Rect2f(cv::Point2f(x1,y1), cv::Point2f(x2,y2)) });
        }
        return true;
    }

    int inputW() const { return H_ > 0 ? W_ : 640; }
    int inputH() const { return H_ > 0 ? H_ : 640; }
    cudaStream_t stream() const { return stream_; }
    void* dInPtr() const { return dIn_; }

private:
    std::unique_ptr<IRuntime, void(*)(IRuntime*)>         runtime_{ nullptr, [](IRuntime* p) { delete p; } };
    std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)>      engine_{ nullptr,  [](ICudaEngine* p) { delete p; } };
    std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)>ctx_{ nullptr,      [](IExecutionContext* p) { delete p; } };

    std::string inName_;
    int N_ = 1, C_ = 3, H_ = 640, W_ = 640;
    void* dIn_ = nullptr;          // half 输入
    cudaStream_t stream_ = nullptr;

    bool hasNMS_ = false;
    std::string outNum_, outBoxes_, outScores_, outClasses_;
    int   maxDet_ = 100;
    void* dNum_ = nullptr; void* dBoxes_ = nullptr; void* dScores_ = nullptr; void* dCls_ = nullptr;
    int* hNum_ = nullptr; float* hBoxes_ = nullptr; float* hScores_ = nullptr; int* hCls_ = nullptr;
};

// ======================= 逻辑参数（固定区域） =======================
static const wchar_t* kWindowTitle = L"Beat World (全网BGP线路）";
static const int LEFT_X = 395;
static const int TOP_Y = 80;
static const int WIDTH_MASK = 240;
static const int HEIGHT_MASK = 595;
static const int REGION_RIGHT = LEFT_X + WIDTH_MASK;
static const int REGION_BOTTOM = TOP_Y + HEIGHT_MASK;
static const int EXTRA_OFFSET = 67;

// 方向标签
static const std::unordered_map<std::string, bool> LEFT_LABEL = {
    {"N7",true},{"N4",true},{"N1",true},{"N9",false},{"N6",false},{"N3",false}
};
static inline std::string HalfMap(const std::string& raw, float cx, float midx) {
    if (cx < midx) { if (raw == "N9")return "N7"; if (raw == "N6")return "N4"; if (raw == "N3")return "N1"; return raw; }
    else { if (raw == "N7")return "N9"; if (raw == "N4")return "N6"; if (raw == "N1")return "N3"; return raw; }
}

// DD 键值映射
static const std::unordered_map<std::string, int> KEY_MAP = {
    {"N7",807},{"N4",804},{"N1",801},{"N9",809},{"N6",806},{"N3",803}
};

// ======================= 阈值配置（从文件读取） =======================
struct KeyThresholds {
    double N7{}, N4{}, N1{}, N9{}, N6{}, N3{}, SPACE{};
    int KEY_DOWN_MS{ 5 };
};

static bool parseDouble(const std::string& s, double& out) {
    try {
        size_t idx = 0;
        out = std::stod(s, &idx);
        if (idx != s.size()) return false;
        if (!std::isfinite(out)) return false;
        return true;
    }
    catch (...) { return false; }
}

static bool loadThresholdsFile(const std::filesystem::path& file, KeyThresholds& thr) {
    std::ifstream fin(file);
    if (!fin.is_open()) {
        std::wcerr << L"[Config] 打不开阈值文件: " << file.wstring() << L"\n";
        return false;
    }

    std::map<std::string, double> kv;
    std::string line; int lineNo = 0;
    while (std::getline(fin, line)) {
        ++lineNo;
        std::string s = trim(line);
        if (s.empty()) continue;
        if (s[0] == '#' || s[0] == ';') continue;

        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            std::cerr << "[Config] 第 " << lineNo << " 行缺少 '=' : " << line << "\n";
            return false;
        }
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        if (key.empty() || val.empty()) {
            std::cerr << "[Config] 第 " << lineNo << " 行键或值为空: " << line << "\n";
            return false;
        }
        toUpperInplace(key);

        double d = 0.0;
        if (!parseDouble(val, d)) {
            std::cerr << "[Config] 第 " << lineNo << " 行解析数值失败: " << line << "\n";
            return false;
        }
        kv[key] = d;
    }

    // 必须包含的键
    const char* required[] = { "N7","N4","N1","N9","N6","N3","SPACE","KEY_DOWN_MS" };
    for (auto k : required) {
        if (!kv.count(k)) {
            std::cerr << "[Config] 缺少阈值键: " << k << "\n";
            return false;
        }
    }

    thr.N7 = kv["N7"];  thr.N4 = kv["N4"];  thr.N1 = kv["N1"];
    thr.N9 = kv["N9"];  thr.N6 = kv["N6"];  thr.N3 = kv["N3"];
    thr.SPACE = kv["SPACE"];
    // 读 KEY_DOWN_MS
    thr.KEY_DOWN_MS = (int)std::lround(std::clamp(kv["KEY_DOWN_MS"], 1.0, 50.0));

    std::cout << std::fixed << std::setprecision(3)
        << "[Config] Loaded thresholds:"
        << " N7=" << thr.N7
        << " N4=" << thr.N4
        << " N1=" << thr.N1
        << " | N9=" << thr.N9
        << " N6=" << thr.N6
        << " N3=" << thr.N3
        << " | SPACE=" << thr.SPACE
        << " | KEY_DOWN_MS=" << std::setprecision(0) << thr.KEY_DOWN_MS << "ms\n";

    return true;
}

static inline double getKeyThreshold(const KeyThresholds& t, const std::string& keyUpper) {
    if (keyUpper == "N7")   return t.N7;
    else if (keyUpper == "N4")   return t.N4;
    else if (keyUpper == "N1")   return t.N1;
    else if (keyUpper == "N9")   return t.N9;
    else if (keyUpper == "N6")   return t.N6;
    else if (keyUpper == "N3")   return t.N3;
    else if (keyUpper == "SPACE")return t.SPACE;
    return std::numeric_limits<double>::quiet_NaN();
}

// ======================= 主程序 =======================
int wmain() {
    CUresult cuerr = cuInit(0);
    if (cuerr != CUDA_SUCCESS) {
        std::cerr << "[CUDA] cuInit failed: " << (int)cuerr << "\n";
        return 1;
    }
    winrt::init_apartment(winrt::apartment_type::single_threaded);
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

    // 读取阈值文件
    KeyThresholds THR{};
    auto thrPath = toExeDir("thresholds.txt");
    if (!loadThresholdsFile(thrPath, THR)) {
        std::cerr << "[Main] 阈值配置错误，程序退出。\n";
        return 5;
    }

    // 找窗口
    HWND hwnd = nullptr;
    {
        auto toLower = [](const std::wstring& s) { std::wstring t = s; for (auto& c : t) c = (wchar_t)towlower(c); return t; };
        struct FD { std::wstring target; HWND exact = nullptr, sub = nullptr; } fd{ toLower(kWindowTitle) };
        EnumWindows([](HWND h, LPARAM lp)->BOOL {
            if (!IsWindowVisible(h)) return TRUE;
            wchar_t buf[512]{}; GetWindowTextW(h, buf, 512); if (!buf[0]) return TRUE;
            auto& fd = *reinterpret_cast<FD*>(lp); auto low = std::wstring(buf); for (auto& c : low) c = (wchar_t)towlower(c);
            if (!fd.exact && low == fd.target) { fd.exact = h; return FALSE; }
            if (!fd.sub && low.find(fd.target) != std::wstring::npos) fd.sub = h;
            return TRUE;
            }, (LPARAM)&fd);
        hwnd = fd.exact ? fd.exact : fd.sub;
    }
    if (!hwnd) { std::wcerr << L"[Main] 找不到窗口: " << kWindowTitle << L"\n"; return 2; }

    // 初始化 WGC
    WGCCapture cap(hwnd);
    if (!cap.init()) { std::cerr << "[Main] WGC 初始化失败\n"; return 3; }
    cap.start();

    // 初始化 DD
    InitDD();

    // 启动异步发键器（从配置读取时长）
    KeyTapSender keySender;
    keySender.start();

    // 初始化 TRT（FP16 输入 + EfficientNMS）
    YoloTRT yolo;
    if (!yolo.init("best08201.engine")) {
        std::cerr << "[Main] TensorRT 初始化失败\n";
        cap.stop(); return 4;
    }

    static const std::vector<std::string> class_names = { "N7","N4","N1","N9","N6","N3","SPACE" };

    double tLastFPS = nowSec();
    int    infCount = 0;
    std::unordered_map<std::string, double> lastTrig;

    while (true) {
        cudaArray_t arr = nullptr;
        int fullW = 0, fullH = 0, bufIdx = -1;
        if (!cap.acquireLatestCudaArray(arr, fullW, fullH, bufIdx)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ROI：下半屏 + EXTRA_OFFSET
        int y_start = clampv(fullH / 2 + EXTRA_OFFSET, 0, std::max(0, fullH - 1));
        int roiX = 0, roiY = y_start, roiW = fullW, roiH = fullH - y_start;
        if (roiH <= 0) { cap.releaseCudaArray(bufIdx); continue; }

        // letterbox 参数（host 端计算）
        float s = std::min(yolo.inputW() / (float)roiW, yolo.inputH() / (float)roiH);
        int new_w = (int)std::round(roiW * s);
        int new_h = (int)std::round(roiH * s);
        int pad_x = (yolo.inputW() - new_w) / 2;
        int pad_y = (yolo.inputH() - new_h) / 2;

        // 把 BGRA 纹理的 ROI letterbox 到 half-CHW
        ppRunLetterboxFromCudaArray(
            arr, fullW, fullH,
            roiX, roiY, roiW, roiH,
            reinterpret_cast<__half*>(yolo.dInPtr()),
            yolo.inputW(), yolo.inputH(),
            s, pad_x, pad_y, 114,
            yolo.stream()
        );

        cap.releaseCudaArray(bufIdx);

        // 推理 + 小结果回拷 + 解析
        std::vector<Det> dets_roi;
        if (!yolo.enqueueAndCollect(0.25f, dets_roi, s, pad_x, pad_y, roiW, roiH)) continue;

        // 每秒打印推理帧率
        infCount++;
        const double nowT = nowSec();
        if (nowT - tLastFPS >= 1.0) {
            double fps = infCount / (nowT - tLastFPS);
            std::cout << std::fixed << "[FPS] INF ~ " << std::setprecision(1) << fps << "/s\n";
            tLastFPS = nowT;
            infCount = 0;
        }

        // 一帧内“每键只选一个最佳候选” + 异步发键
        struct Cand { int ddcode; double thr; float cxg; float delta; };
        std::unordered_map<std::string, Cand> best;

        for (const auto& d : dets_roi) {
            float gx1 = d.box.x;
            float gy1 = d.box.y + (float)roiY;
            float gx2 = d.box.x + d.box.width;
            float gy2 = d.box.y + d.box.height + (float)roiY;
            float cxg = 0.5f * (gx1 + gx2);
            float cyg = 0.5f * (gy1 + gy2);

            std::string raw = (d.cls >= 0 && d.cls < (int)class_names.size()) ? class_names[d.cls] : "Unknown";
            if (raw == "Unknown") continue;

            std::string mapped = HalfMap(raw, cxg, (float)fullW * 0.5f);
            if (mapped == "SPACE") continue;

            // 屏蔽区
            if (cxg >= LEFT_X && cxg <= REGION_RIGHT && cyg >= TOP_Y && cyg <= REGION_BOTTOM) continue;

            auto itKey = KEY_MAP.find(mapped); if (itKey == KEY_MAP.end()) continue;
            bool isLeft = LEFT_LABEL.count(mapped) ? LEFT_LABEL.at(mapped) : false;
            double thr = getKeyThreshold(THR, mapped);
            if (!std::isfinite(thr)) continue;

            float delta = isLeft ? (cxg - (float)thr) : ((float)thr - cxg);
            if (delta < 0.f) continue; // 还没越线

            auto it = best.find(mapped);
            if (it == best.end() || delta < it->second.delta) {
                best[mapped] = Cand{ itKey->second, thr, cxg, delta };
            }
        }

        for (auto& kv : best) {
            const std::string& mapped = kv.first;
            const Cand& c = kv.second;

            // 去抖：100ms
            auto itLT = lastTrig.find(mapped);
            if (itLT != lastTrig.end() && (nowT - itLT->second < 0.05)) continue;

            keySender.enqueue(c.ddcode, THR.KEY_DOWN_MS);
            lastTrig[mapped] = nowT;

            float signedDelta = (LEFT_LABEL.at(mapped) ? (c.cxg - (float)c.thr) : ((float)c.thr - c.cxg));
            if (std::fabs(signedDelta) > 5.f) {
                std::cout << std::fixed << "[Trig] " << mapped
                    << " X=" << std::setprecision(1) << c.cxg
                    << " Thr=" << std::setprecision(1) << c.thr
                    << " Δ=" << (signedDelta >= 0 ? "+" : "") << std::setprecision(1) << signedDelta
                    << " |Q=" << keySender.depth()
                    << "\n";
            }
        }
    }

    // 如果后续加退出条件，别忘了：
    // keySender.stop();
}   