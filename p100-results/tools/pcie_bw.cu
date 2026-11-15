// pcie_bw.cu - pinned host<->device copy bandwidth for one GPU.
// Run it under numactl to choose where the host buffer lives, e.g.
//   numactl --membind=0 --cpunodebind=0 ./pcie_bw 0     (GPU 0, host memory on node 0: local)
//   numactl --membind=1 --cpunodebind=1 ./pcie_bw 0     (GPU 0, host memory on node 1: across the QPI link)
// Prints H2D and D2H GB/s (1e9 bytes/s) as the median of N timed copies after a warm-up.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t mib = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    const size_t bytes = mib << 20;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    void* h = nullptr;
    void* d = nullptr;
    CK(cudaHostAlloc(&h, bytes, cudaHostAllocDefault));
    std::memset(h, 1, bytes);  // first touch: places the pages on the node numactl chose
    CK(cudaMalloc(&d, bytes));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    auto run = [&](bool h2d) {
        std::vector<double> gbs;
        for (int i = 0; i < reps + 2; ++i) {
            CK(cudaEventRecord(a, s));
            if (h2d) CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s));
            else CK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s));
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            if (i >= 2) gbs.push_back((double) bytes / (ms * 1e-3) / 1e9);
        }
        return median(gbs);
    };
    const double h2d = run(true), d2h = run(false);
    std::printf("gpu %d (%s, pci %02x:%02x.0) buffer %zu MiB x %d: H2D %.2f GB/s  D2H %.2f GB/s\n", dev, p.name, p.pciBusID,
                p.pciDeviceID, mib, reps, h2d, d2h);
    CK(cudaFree(d));
    CK(cudaFreeHost(h));
    return 0;
}
