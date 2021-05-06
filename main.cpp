#include <array>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wincodec.h>

#include <fmt\format.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace Microsoft::WRL;

template <typename T>
struct InlineData
{
    T data;
    InlineData(T data) : data(data) { }
    T* operator*() { return &data; }
};

template <typename T, size_t N>
struct InlineArray
{
    T data[N];
    constexpr InlineArray(T values[N])
    {
        std::copy_n(values, N, data);
    }
    constexpr InlineArray(std::initializer_list<T> list)
    {
        std::copy_n(list.begin(), N, data);
    }
    constexpr InlineArray(T first, ...)
    {
        va_list args;
        va_start(args, first);
        for (int i = 0; i < first; i++)
            data[i] = va_arg(args, T); 
        va_end(args);
    }
    constexpr T* operator*() { return data; }
};

size_t aligned_size(size_t sz, size_t alignment)
{
    return ((sz - 1) & alignment) + alignment;
}
template <typename T>
size_t aligned_size(const std::vector<T>& v, size_t alignment)
{
    return aligned_size(sizeof(T) * T.size(), alignment);
}
template <typename T, size_t N>
size_t aligned_size(const std::array<T,N>& v, size_t alignment)
{
    return aligned_size(sizeof(T) * N, alignment);
}

struct vertex_t
{
    glm::vec3 pos;
    glm::vec2 tex;
    vertex_t() = default;
    vertex_t(glm::vec3 pos, glm::vec2 tex) : pos(pos), tex(tex) { }
};

struct vertex_uniform_t
{
    glm::mat4 model;
    vertex_uniform_t() : model(glm::identity<glm::mat4>()) { }
};

struct pixel_uniform_t
{
    glm::vec4 color;
    pixel_uniform_t() : color(1, 0, 0, 1) { }
};

template <typename T>
void debug_name(const ComPtr<T>& com, const std::wstring& name)
{
    ComPtr<ID3D12Object> obj;
    if (SUCCEEDED(com.As(&obj)))
        obj->SetName(name.c_str());
}

#define TRY(I) if (!SUCCEEDED(I)) { \
    std::cout << "Failed to execute "###I << std::endl;\
    throw std::runtime_error("Failed to execute "###I);\
}

std::tuple<ComPtr<ID3D12Device>, DXGI_ADAPTER_DESC>  find_device()
{
    ComPtr<IDXGIFactory2> factory;
    TRY(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    for (uint32_t i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC desc;
        TRY(adapter->GetDesc(&desc));
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device))))
        {
//             if (D3D12_FEATURE_DATA_D3D12_OPTIONS5 features;
//                 SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features, sizeof(features))) &&
//                 features.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0)
            {
                std::wcout << "RayTracing Adapter: " << desc.Description << std::endl;
                return { device, desc };
            }
        }
    }
    throw std::runtime_error("No suitable devices found");
}

bool enable_debug_layer()
{
    ComPtr<ID3D12Debug> debug;
    ComPtr<ID3D12Debug1> debug1;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))) &&
        SUCCEEDED(debug.As(&debug1)))
    {
        debug1->EnableDebugLayer();
        debug1->SetEnableGPUBasedValidation(true);
        return true;
    }
    return false;
}

LRESULT CALLBACK wnd_proc(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_CREATE:
        return 0; // Everything is fine, continue with creation.
    case WM_CLOSE:
        PostQuitMessage(EXIT_SUCCESS);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wp, lp);
}

HWND create_window(int width, int height)
{
    WNDCLASS wc{
        .style         = CS_HREDRAW | CS_VREDRAW,
        .lpfnWndProc   = wnd_proc,
        .cbClsExtra    = 0,
        .cbWndExtra    = 0,
        .hInstance     = GetModuleHandle(nullptr),
        .hIcon         = LoadIcon(nullptr, IDI_APPLICATION),
        .hCursor       = LoadCursor(nullptr, IDC_ARROW),
        .hbrBackground = GetSysColorBrush(COLOR_WINDOW),
        .lpszMenuName  = nullptr,
        .lpszClassName = TEXT("D3D12window"),
    };
    if (!RegisterClass(&wc))
        return NULL;
    RECT r = { 0, 0, width, height };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false);
    return CreateWindow(wc.lpszClassName, TEXT("D3D12"), 
        WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
        r.right - r.left, r.bottom - r.top, NULL, NULL, wc.hInstance, nullptr);
}

ComPtr<IDXGISwapChain4> create_swapchain(const ComPtr<ID3D12CommandQueue>& cmd, HWND hWnd, int width, int height)
{
    ComPtr<IDXGIFactory2> factory;
    TRY(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGISwapChain1> swapchain;
    DXGI_SWAP_CHAIN_DESC1 desc
    {
        .Width       = (UINT)width,
        .Height      = (UINT)height,
        .Format      = DXGI_FORMAT_B8G8R8A8_UNORM,
        .Stereo      = false,
        .SampleDesc  = { 1, 0 },
        .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
        .BufferCount = 2,
        .Scaling     = DXGI_SCALING_STRETCH,
        .SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD,
        .AlphaMode   = DXGI_ALPHA_MODE_UNSPECIFIED,
        .Flags       = 0,
    };
    TRY(factory->CreateSwapChainForHwnd(cmd.Get(), hWnd, &desc, nullptr, nullptr, &swapchain));
    ComPtr<IDXGISwapChain4> swapchain4;
    TRY(swapchain.As(&swapchain4));
    return swapchain4;
}

std::vector<uint8_t> load_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (ifs.is_open())
    {
        size_t size = ifs.tellg();
        std::vector<uint8_t> buffer(size);
        ifs.seekg(std::ios::beg);
        ifs.read(reinterpret_cast<char*>(buffer.data()), size);
        return buffer;
    }
    return {};
}


std::tuple<glm::uvec2, std::vector<uint8_t>> load_image(const std::string& path)
{
    try
    {
        CoInitialize(0);
        int wpath_size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
            (uint32_t)path.size(), nullptr, 0);
        std::wstring wpath(wpath_size, 0);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (uint32_t)path.size(),
            wpath.data(), (uint32_t)wpath.size());

        ComPtr<IDXGIFactory> factory;
        TRY(CreateDXGIFactory(IID_PPV_ARGS(&factory)));

        ComPtr<IWICImagingFactory> wic_factory;
        TRY(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
            CLSCTX_ALL, IID_PPV_ARGS(&wic_factory)));

        ComPtr<IEnumUnknown> enumerator;
        TRY(wic_factory->CreateComponentEnumerator(WICComponentType::WICEncoder,
            WICComponentEnumerateOptions::WICComponentEnumerateDefault, &enumerator));
        ComPtr<IUnknown> item;
        ULONG items;
        while (enumerator->Next(1, &item, &items) == S_OK)
        {
            ComPtr<IWICBitmapEncoderInfo> info;
            item.As(&info);
            UINT items;
            std::wstring name(256, 0);
            info->GetFriendlyName(name.size(), name.data(), &items);
            std::wcout << "codec: " << name << std::endl;
        }

        ComPtr<IWICBitmapDecoder> wic_decoder;
        TRY(wic_factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, &wic_decoder));
        uint32_t image_frames = 0;
        if (SUCCEEDED(wic_decoder->GetFrameCount(&image_frames)) && image_frames > 0)
        {
            ComPtr<IWICBitmapFrameDecode> frame;
            TRY(wic_decoder->GetFrame(0, &frame));
            glm::uvec2 size;
            frame->GetSize(&size.x, &size.y);
            if (size.x == 0 || size.y == 0)
                throw std::runtime_error("Invalid image size");
            ComPtr<IWICFormatConverter> converter;
            TRY(wic_factory->CreateFormatConverter(&converter));
            TRY(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
                WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom));
            uint32_t stride = size.x * 4;
            uint32_t bytes = stride * size.y;
            std::vector<uint8_t> image_data(bytes);
            TRY(converter->CopyPixels(nullptr, stride, (uint32_t)image_data.size(), image_data.data()));
            return { size, image_data };
        }
        else
        {
            throw std::runtime_error("Could not fetch image data");
        }
    }
    catch (const std::exception&)
    {
        std::cout << "Failed to load image " << path << std::endl;
        return {};
    }
}

int main()
{
    HWND hWnd = create_window(800, 600);
    enable_debug_layer();
    auto [device, device_info] = find_device();
    debug_name(device, TEXT("device"));

    std::wcout << device_info.Description << "\n";

    if (ComPtr<ID3D12InfoQueue> info_queue;
        SUCCEEDED(device.As(&info_queue)))
    {
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_MESSAGE, true);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_INFO, true);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
    }

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc;
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;
    TRY(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)));
    debug_name(queue, TEXT("main queue"));

    ComPtr<IDXGISwapChain4> swapchain = create_swapchain(queue, hWnd, 800, 600);
    debug_name(swapchain, TEXT("swapchain"));

    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc;
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 2;
    rtv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtv_heap_desc.NodeMask = 0;
    TRY(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap)));
    debug_name(rtv_heap, TEXT("rtv_heap"));

    ComPtr<ID3D12DescriptorHeap> srv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc;
    srv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srv_heap_desc.NumDescriptors = 3;
    srv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    srv_heap_desc.NodeMask = 0;
    TRY(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap)));
    debug_name(srv_heap, TEXT("srv_heap"));

    D3D12_CPU_DESCRIPTOR_HANDLE rtv_heap_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srv_heap_handle = srv_heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_GPU_DESCRIPTOR_HANDLE rtv_heap_handle_gpu = rtv_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE srv_heap_handle_gpu = srv_heap->GetGPUDescriptorHandleForHeapStart();

    UINT rtv_heap_handle_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    UINT srv_heap_handle_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    ComPtr<ID3D12CommandAllocator> cmd_pool;
    TRY(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmd_pool)));
    debug_name(cmd_pool, TEXT("cmd_pool"));

    std::array root_ranges
    {
        // buffer vertex
        D3D12_DESCRIPTOR_RANGE
        {
            .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
            .NumDescriptors = 1,
            .BaseShaderRegister = 1,
            .RegisterSpace = 0,
            .OffsetInDescriptorsFromTableStart = 0,
        },
        // buffer pixel
        D3D12_DESCRIPTOR_RANGE
        {
            .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
            .NumDescriptors = 1,
            .BaseShaderRegister = 0,
            .RegisterSpace = 0,
            .OffsetInDescriptorsFromTableStart = 1,
        },
        D3D12_DESCRIPTOR_RANGE
        {
            .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            .NumDescriptors = 1,
            .BaseShaderRegister = 0,
            .RegisterSpace = 0,
            .OffsetInDescriptorsFromTableStart = 2,
        },
    };
    std::array root_params
    {
        D3D12_ROOT_PARAMETER
        {
            .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            .DescriptorTable = { root_ranges.size(), root_ranges.data() },
            .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
        },
    };

    D3D12_STATIC_SAMPLER_DESC sampler_desc{};
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.MipLODBias = 0;
    sampler_desc.MaxAnisotropy = 0;
    sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler_desc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc.ShaderRegister = 0;
    sampler_desc.RegisterSpace = 0;
    sampler_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootsig_desc{};
    rootsig_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    rootsig_desc.NumParameters = (UINT)root_params.size();
    rootsig_desc.pParameters = root_params.data();
    rootsig_desc.NumStaticSamplers = 1;
    rootsig_desc.pStaticSamplers = &sampler_desc;

    ComPtr<ID3DBlob> rootsig_blob;
    TRY(D3D12SerializeRootSignature(&rootsig_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, 
        &rootsig_blob, nullptr));

    ComPtr<ID3D12RootSignature> rootsig;
    TRY(device->CreateRootSignature(0, rootsig_blob->GetBufferPointer(),
        rootsig_blob->GetBufferSize(), IID_PPV_ARGS(&rootsig)));

    std::vector<uint8_t> pixel_shader_bytecode = load_file("PixelShader.cso");
    std::vector<uint8_t> vertex_shader_bytecode = load_file("VertexShader.cso");

    std::array<D3D12_INPUT_ELEMENT_DESC, 2> input_element_desc{
        D3D12_INPUT_ELEMENT_DESC{ "POS", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        D3D12_INPUT_ELEMENT_DESC{ "TEX", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_desc{};
    pipeline_desc.pRootSignature = rootsig.Get();
    pipeline_desc.VS = { vertex_shader_bytecode.data(), (UINT)vertex_shader_bytecode.size() };
    pipeline_desc.PS = { pixel_shader_bytecode.data(), (UINT)pixel_shader_bytecode.size() };
    pipeline_desc.BlendState.AlphaToCoverageEnable = false;
    pipeline_desc.BlendState.IndependentBlendEnable = false;
    pipeline_desc.BlendState.RenderTarget[0].BlendEnable = false;
    pipeline_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pipeline_desc.SampleMask = UINT_MAX;
    pipeline_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipeline_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pipeline_desc.DepthStencilState.DepthEnable = false;
    pipeline_desc.DepthStencilState.StencilEnable = false;
    pipeline_desc.InputLayout.NumElements = (UINT)input_element_desc.size();
    pipeline_desc.InputLayout.pInputElementDescs = input_element_desc.data();
    pipeline_desc.IBStripCutValue;
    pipeline_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_desc.NumRenderTargets = 1;
    pipeline_desc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    pipeline_desc.DSVFormat;
    pipeline_desc.SampleDesc = { 1, 0 };
    pipeline_desc.NodeMask = 0;
    pipeline_desc.CachedPSO;
    pipeline_desc.Flags;
    ComPtr<ID3D12PipelineState> pipeline;
    TRY(device->CreateGraphicsPipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline)));
    
    vertex_uniform_t vertex_uniform;
    pixel_uniform_t pixel_uniform;
    std::array<vertex_t, 4> quad_data{
        vertex_t({ -0.5f, -0.5f,  0.0f }, { 0.0f, 0.0f }),
        vertex_t({ -0.5f,  0.5f,  0.0f }, { 0.0f, 1.0f }),
        vertex_t({  0.5f,  0.5f,  0.0f }, { 1.0f, 1.0f }),
        vertex_t({  0.5f, -0.5f,  0.0f }, { 1.0f, 0.0f }),
    };
    std::array<uint32_t, 6> quad_indices{ 0, 1, 2, 0, 2, 3 };
    size_t quad_vertices_size = aligned_size(quad_data, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    size_t quad_indices_size = aligned_size(quad_indices, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    size_t vertex_uniform_size = aligned_size(sizeof(vertex_uniform_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    size_t pixel_uniform_size = aligned_size(sizeof(pixel_uniform_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    auto [image_size, image_data] = load_image("image.png");
    size_t image_data_size = image_size.x * image_size.y * 4;

    D3D12_HEAP_PROPERTIES image_heap_props{};
    image_heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC image_buffer_resource_desc{};
    image_buffer_resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image_buffer_resource_desc.Width = image_size.x;
    image_buffer_resource_desc.Height = image_size.y;
    image_buffer_resource_desc.DepthOrArraySize = 1;
    image_buffer_resource_desc.MipLevels = 1;
    image_buffer_resource_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    image_buffer_resource_desc.SampleDesc = { 1, 0 };
    image_buffer_resource_desc.Layout = {};
    
    ComPtr<ID3D12Resource> image_texture;
    TRY(device->CreateCommittedResource(&image_heap_props, D3D12_HEAP_FLAG_NONE,
        &image_buffer_resource_desc, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&image_texture)));

    UINT64 image_upload_size = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    device->GetCopyableFootprints(&image_buffer_resource_desc, 0, 1, 0, &footprint, nullptr, nullptr, &image_upload_size);

    image_heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    image_buffer_resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    image_buffer_resource_desc.Width = image_upload_size;
    image_buffer_resource_desc.Height = 1;
    image_buffer_resource_desc.DepthOrArraySize = 1;
    image_buffer_resource_desc.MipLevels = 1;
    image_buffer_resource_desc.Format = DXGI_FORMAT_UNKNOWN;
    image_buffer_resource_desc.SampleDesc = { 1, 0 };
    image_buffer_resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    
    ComPtr<ID3D12Resource> image_buffer;
    TRY(device->CreateCommittedResource(&image_heap_props, D3D12_HEAP_FLAG_NONE,
        &image_buffer_resource_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&image_buffer)));

    if (void* ptr; SUCCEEDED(image_buffer->Map(0, nullptr, &ptr)))
    {
        for (UINT y = 0; y < image_size.y; y++)
        {
            std::copy_n(image_data.data() + y * image_size.x * 4, image_size.x * 4,
                static_cast<uint8_t*>(ptr) + y * footprint.Footprint.RowPitch);
        }
        image_buffer->Unmap(0, nullptr);
    }

    ComPtr<ID3D12GraphicsCommandList> cmd_image;
    TRY(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmd_pool.Get(),
        nullptr, IID_PPV_ARGS(&cmd_image)));
    {
        D3D12_TEXTURE_COPY_LOCATION copy_src;
        copy_src.pResource = image_buffer.Get();
        copy_src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        copy_src.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION copy_dst;
        copy_dst.pResource = image_texture.Get();
        copy_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        copy_dst.SubresourceIndex = 0;
        cmd_image->CopyTextureRegion(&copy_dst, 0, 0, 0, &copy_src, nullptr);
        D3D12_RESOURCE_BARRIER barrier;
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = image_texture.Get();
        barrier.Transition.Subresource = 0;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmd_image->ResourceBarrier(1, &barrier);
        cmd_image->Close();
    }


    ID3D12CommandList* image_cmd_list = cmd_image.Get();
    queue->ExecuteCommandLists(1, &image_cmd_list);

    ComPtr<ID3D12Fence> image_fence;
    TRY(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&image_fence)));
    debug_name(image_fence, TEXT("upload image fence"));

    if (HANDLE image_upload_event = CreateEvent(nullptr, false, false, TEXT("ImageUploadEvent"));
        SUCCEEDED(image_upload_event) &&
        SUCCEEDED(image_fence->SetEventOnCompletion(1, image_upload_event)))
    {
        TRY(queue->Signal(image_fence.Get(), 1));
        DWORD result = WaitForSingleObject(image_upload_event, INFINITE);
        assert(result == WAIT_OBJECT_0);
    }

    image_buffer = nullptr;
    
    //if (void* ptr; SUCCEEDED(image_buffer->Map(0, nullptr, &ptr)))
    //{
    //    uint8_t* dst = static_cast<uint8_t*>(ptr);
    //    std::copy(image_data.begin(), image_data.end(), dst);
    //    image_buffer->Unmap(0, nullptr);
    //}

    ComPtr<ID3D12Resource> buffer;
    
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC buffer_resource_desc{};
    buffer_resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_resource_desc.Width = quad_vertices_size + quad_indices_size + 
        vertex_uniform_size + pixel_uniform_size;
    buffer_resource_desc.Height = 1;
    buffer_resource_desc.DepthOrArraySize = 1;
    buffer_resource_desc.MipLevels = 1;
    buffer_resource_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_resource_desc.SampleDesc = { 1, 0 };
    buffer_resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    TRY(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE,
        &buffer_resource_desc, D3D12_RESOURCE_STATE_GENERIC_READ, 
        nullptr, IID_PPV_ARGS(&buffer)));

    if (void* ptr; SUCCEEDED(buffer->Map(0, nullptr, &ptr)))
    {
        auto origin = ptr;
        size_t space = buffer_resource_desc.Width;
        std::align(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
            sizeof(vertex_t)* quad_data.size(), ptr, space);
        std::copy(quad_data.begin(), quad_data.end(), reinterpret_cast<vertex_t*>(ptr));
        ptr = (void*)((uint8_t*)ptr + sizeof(vertex_t) * quad_data.size());
        
        std::align(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
            sizeof(uint32_t)* quad_indices.size(), ptr, space);
        std::copy(quad_indices.begin(), quad_indices.end(), reinterpret_cast<uint32_t*>(ptr));
        ptr = (void*)((uint8_t*)ptr + sizeof(uint32_t) * quad_indices.size());

        std::align(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
            sizeof(vertex_uniform_t), ptr, space);
        *reinterpret_cast<vertex_uniform_t*>(ptr) = vertex_uniform;
        ptr = (void*)((uint8_t*)ptr + sizeof(vertex_uniform_t) * quad_indices.size());

        std::align(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
            sizeof(vertex_uniform_t), ptr, space);
        *reinterpret_cast<pixel_uniform_t*>(ptr) = pixel_uniform;
        ptr = (void*)((uint8_t*)ptr + sizeof(pixel_uniform_t) * quad_indices.size());

        buffer->Unmap(0, nullptr);
    }

    D3D12_VERTEX_BUFFER_VIEW vertex_buffer_view{};
    vertex_buffer_view.BufferLocation = buffer->GetGPUVirtualAddress();
    vertex_buffer_view.SizeInBytes = sizeof(vertex_t) * quad_data.size();
    vertex_buffer_view.StrideInBytes = sizeof(vertex_t);

    D3D12_INDEX_BUFFER_VIEW index_buffer_view{};
    index_buffer_view.BufferLocation = vertex_buffer_view.BufferLocation + quad_vertices_size;
    index_buffer_view.SizeInBytes = sizeof(uint32_t) * quad_indices.size();
    index_buffer_view.Format = DXGI_FORMAT_R32_UINT;

    D3D12_CONSTANT_BUFFER_VIEW_DESC vertex_uniform_view{};
    vertex_uniform_view.BufferLocation = index_buffer_view.BufferLocation + quad_indices_size;
    vertex_uniform_view.SizeInBytes = aligned_size(sizeof(vertex_uniform_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    device->CreateConstantBufferView(&vertex_uniform_view, srv_heap_handle);

    D3D12_CONSTANT_BUFFER_VIEW_DESC pixel_uniform_view{};
    pixel_uniform_view.BufferLocation = vertex_uniform_view.BufferLocation + vertex_uniform_size;
    pixel_uniform_view.SizeInBytes = aligned_size(sizeof(pixel_uniform_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    device->CreateConstantBufferView(&pixel_uniform_view, { srv_heap_handle.ptr + srv_heap_handle_size });

    D3D12_SHADER_RESOURCE_VIEW_DESC image_texture_view{};
    image_texture_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    image_texture_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    image_texture_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    image_texture_view.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(image_texture.Get(), &image_texture_view, { srv_heap_handle.ptr + srv_heap_handle_size * 2 });

    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> swapchain_rtv_handle(2);
    std::vector<ComPtr<ID3D12Resource>> swapchain_buffers(2);
    std::vector<ComPtr<ID3D12GraphicsCommandList>> cmd_list(2);
    for (UINT buffer_index = 0; buffer_index < 2; buffer_index++)
    {
        TRY(swapchain->GetBuffer(buffer_index, IID_PPV_ARGS(&swapchain_buffers[buffer_index])));
        swapchain_rtv_handle[buffer_index] = { rtv_heap_handle.ptr + rtv_heap_handle_size * buffer_index };
        device->CreateRenderTargetView(swapchain_buffers[buffer_index].Get(), 0, swapchain_rtv_handle[buffer_index]);
        debug_name(swapchain_buffers[buffer_index], fmt::format(TEXT("swapchain_buffers#{}"), buffer_index));

        TRY(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmd_pool.Get(),
            pipeline.Get(), IID_PPV_ARGS(&cmd_list[buffer_index])));
        {
            debug_name(cmd_list[buffer_index], fmt::format(TEXT("cmd_list#{}"), buffer_index));
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = swapchain_buffers[buffer_index].Get();
            barrier.Transition.Subresource = 0;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            cmd_list[buffer_index]->ResourceBarrier(1, &barrier);

            cmd_list[buffer_index]->ClearRenderTargetView(swapchain_rtv_handle[buffer_index],
                *InlineArray<float, 4>({ 1, 0, 0, 1 }), 0, nullptr);

            cmd_list[buffer_index]->SetGraphicsRootSignature(rootsig.Get());
            cmd_list[buffer_index]->SetDescriptorHeaps(1, srv_heap.GetAddressOf());
            cmd_list[buffer_index]->SetGraphicsRootDescriptorTable(0, srv_heap->GetGPUDescriptorHandleForHeapStart());
            //cmd_list[buffer_index]->SetGraphicsRootDescriptorTable(1, srv_heap->GetGPUDescriptorHandleForHeapStart());
            
            cmd_list[buffer_index]->OMSetRenderTargets(1, &swapchain_rtv_handle[buffer_index], false, nullptr);
            
            cmd_list[buffer_index]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd_list[buffer_index]->IASetVertexBuffers(0, 1, &vertex_buffer_view);
            cmd_list[buffer_index]->IASetIndexBuffer(&index_buffer_view);
            
            D3D12_VIEWPORT vp = { 0, 0, 800, 600, 0, 1 };
            cmd_list[buffer_index]->RSSetViewports(1, &vp);
            D3D12_RECT scissor = { 0, 0, 800, 600 };
            cmd_list[buffer_index]->RSSetScissorRects(1, &scissor);

            cmd_list[buffer_index]->DrawIndexedInstanced(quad_indices.size(), 1, 0, 0, 0);

            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            cmd_list[buffer_index]->ResourceBarrier(1, &barrier);
        }
        cmd_list[buffer_index]->Close();
    }

    ComPtr<ID3D12Fence> fence;
    TRY(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    debug_name(fence, TEXT("fence"));

    std::array<HANDLE, 2> events{
        CreateEvent(nullptr, false, false, TEXT("FenceEvent1")),
        CreateEvent(nullptr, false, false, TEXT("FenceEvent2")),
    };

//     DWORD thread_id;
//     HANDLE thread_handle = CreateThread(NULL, 0, signal_waiter, &event, 0, &thread_id);
//     if (thread_handle == NULL)
//         std::cout << "Thread creation failed because of error: " << GetLastError() << std::endl;

    auto last_t = std::chrono::high_resolution_clock::now();
    float fps_timer = 0;
    int fps_frames = 0;
    uint64_t fence_value = 0;
    std::array<uint64_t, 2> fence_inflight{ 0, 0 };
    while (true)
    {
        MSG msg;
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto t = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(t - last_t).count();
        last_t = t;

        fps_timer += dt;
        if (fps_timer >= 1.f)
        {
            std::wstring title = fmt::format(TEXT("D3D12 {} - {}fps"), device_info.Description, fps_frames);
            SetWindowText(hWnd, title.c_str());
            fps_timer -= 1.f;
            fps_frames = 0;
        }

        // Render frame

        static float theta = 0.f;
        theta += .01;
        vertex_uniform.model = glm::eulerAngleZ(glm::radians(theta));
        pixel_uniform.color = glm::vec4(glm::abs(glm::sin(theta)), 1, 1, 1);

        D3D12_RANGE buffer_range;
        buffer_range.Begin = quad_vertices_size + quad_indices_size;
        buffer_range.End = buffer_range.Begin + vertex_uniform_size + pixel_uniform_size;
        if (void* ptr; SUCCEEDED(buffer->Map(0, &buffer_range, &ptr)))
        {
            ptr = (void*)((uint8_t*)ptr + quad_vertices_size + quad_indices_size);
            *reinterpret_cast<vertex_uniform_t*>(ptr) = vertex_uniform;

            ptr = (void*)((uint8_t*)ptr + vertex_uniform_size);
            *reinterpret_cast<pixel_uniform_t*>(ptr) = pixel_uniform;

            buffer->Unmap(0, &buffer_range);
        }

        UINT buffer_index = swapchain->GetCurrentBackBufferIndex();
        while (fence->GetCompletedValue() < fence_inflight[buffer_index])
        {
            //std::cout << "wait fence " << buffer_index << "\n";
            fence->SetEventOnCompletion(fence_inflight[buffer_index], events[buffer_index]);
            WaitForSingleObject(events[buffer_index], INFINITE);
        }

        std::array<ID3D12CommandList*, 1> commands{ cmd_list[buffer_index].Get() };
        queue->ExecuteCommandLists(1, commands.data());
        queue->Signal(fence.Get(), ++fence_value);
        fence_inflight[buffer_index] = fence_value;

        swapchain->Present(2, 0);
        fps_frames++;
    }

    queue->Signal(fence.Get(), ++fence_value);
    HANDLE flush_event = CreateEvent(nullptr, true, false, TEXT("FlushEvent"));
    fence->SetEventOnCompletion(fence_value, flush_event);
    WaitForSingleObject(flush_event, INFINITE);

    return EXIT_SUCCESS;
}
