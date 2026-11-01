// src/kernels/cuda/q8_sm60.cuh - a Q8_0 decode GEMV for Pascal GP100 (compute capability 6.0), opt-in with
// STRATA_Q8_SM60=1.  Included by native_mmvq.cu, and by a standalone parity/speed check, so both run this code.
//
// Reads the packed planes of STRATA_Q8_PACKED (the SAME int8 values row-major, n_in bytes per row, then one fp16 scale
// per 32 of them) for every Q8_0 dense matrix and the head, whatever their shape.  Why the packed kernel above is slow
// on a GP100 die (measured on PH402, 1050 MHz: ~137 GB/s for the dense projections, ~160 for the head, against
// 220-270 for this layout): one workgroup per row, eight bytes per thread per pass and a __syncthreads() per row - a
// 2560-wide row is 80 blocks, so the reduction costs more than the row - and the window's activations read as single
// ints, because a q8_1 block is 36 bytes and its values start at byte 4.
//
// Here:
//   * a workgroup first copies the window's activations into shared memory as an int8 plane per column and a float
//     scale per 32 (once; the workgroup then loops over rows), so every activation read is a 16-byte shared load;
//   * TPR threads own a row (a whole warp for n_in >= 2048, a half or a quarter for narrower rows) and read its
//     weights 16 bytes at a time through the read-only cache; the dot of 4 ints is STRATA_DP4A (vmad on sm_60);
//   * the sum is a shuffle within those TPR lanes - no shared memory, no barrier per row.
// The per-32 products are added in another order than the exact kernels', so outputs are not bitwise theirs (float
// rounding only; the int8 dots are exact).
#pragma once

namespace q8sm60 {

constexpr int THREADS = 256;
constexpr std::size_t MAX_SMEM = 48 * 1024;   // a Pascal workgroup's static + dynamic limit
constexpr int MAX_COLS = 8;                        // columns of one launch (a window of 9..16 rows runs in passes)

inline std::size_t smem_bytes(int ncols, int n_in) {
    return std::size_t(ncols) * std::size_t(n_in) + std::size_t(ncols) * std::size_t(n_in / 32) * sizeof(float);
}

template<int NCOLS, int TPR>
__launch_bounds__(THREADS)
__global__ void kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl, const Q81Block* __restrict__ x,
                       float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int4 q8sm60_smem[];
    int8_t* sx = reinterpret_cast<int8_t*>(q8sm60_smem);                         // [NCOLS][n_in]
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * n_in);       // [NCOLS][n_in / 32]
    const int nb = n_in / 32;
    for (int b = int(threadIdx.x); b < NCOLS * nb; b += THREADS) {               // column j = b / nb, block b % nb
        const Q81Block* xb = x + b;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = src[i];
        sd[b] = __low2float(xb->ds);
    }
    __syncthreads();

    constexpr int RPB = THREADS / TPR;
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;
    const int nch = n_in / 16;                                                   // 16-byte chunks, two per block
    const int groups = (n_out + RPB - 1) / RPB;
    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * RPB + rloc;
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        if (row < n_out) {
            const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
            const half* d = dpl + std::size_t(row) * nb;
            for (int c = lane; c < nch; c += TPR) {
                const int4 wv = __ldg(w + c);
                const float dw = __half2float(d[c >> 1]);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * n_in)[c];
                    int s = STRATA_DP4A(wv.x, xv.x, 0);
                    s = STRATA_DP4A(wv.y, xv.y, s);
                    s = STRATA_DP4A(wv.z, xv.z, s);
                    s = STRATA_DP4A(wv.w, xv.w, s);
                    acc[j] += dw * sd[j * nb + (c >> 1)] * float(s);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {                                       // every lane takes part, row or not
            float v = acc[j];
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
            if (lane == 0 && row < n_out) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

// One workgroup per SM would leave too few rows in flight on a 48-SM die; four per SM is what the shapes of
// Qwen3.8-Flash-Next measured best (STRATA_Q8_SM60_BPS overrides it).
inline int blocks_per_sm() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_Q8_SM60_BPS");
        const int n = e ? std::atoi(e) : 0;
        return n > 0 ? n : 4;
    }();
    return v;
}

inline int sm_count() {
    int dev = 0, n = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return 48;
    return n > 0 ? n : 48;
}

template<int NCOLS, int TPR>
void launch_one(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    constexpr int RPB = THREADS / TPR;
    const int groups = (n_out + RPB - 1) / RPB;
    const int grid = (std::min)(groups, sm_count() * blocks_per_sm());
    kernel<NCOLS, TPR><<<unsigned(grid), THREADS, smem_bytes(NCOLS, n_in), s>>>(qs, d, x, y, n_in, n_out);
}

template<int NCOLS>
void launch_cols(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (n_in >= 2048) launch_one<NCOLS, 32>(qs, d, x, y, n_in, n_out, s);
    else if (n_in >= 1024) launch_one<NCOLS, 16>(qs, d, x, y, n_in, n_out, s);
    else launch_one<NCOLS, 8>(qs, d, x, y, n_in, n_out, s);
}

// false: a shape this kernel does not take (the caller runs its usual kernel)
inline bool launch(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                   cudaStream_t s) {
    if (n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS || smem_bytes(ncols, n_in) > MAX_SMEM) return false;
    switch (ncols) {
        case 1: launch_cols<1>(qs, d, x, y, n_in, n_out, s); break;
        case 2: launch_cols<2>(qs, d, x, y, n_in, n_out, s); break;
        case 3: launch_cols<3>(qs, d, x, y, n_in, n_out, s); break;
        case 4: launch_cols<4>(qs, d, x, y, n_in, n_out, s); break;
        case 5: launch_cols<5>(qs, d, x, y, n_in, n_out, s); break;
        case 6: launch_cols<6>(qs, d, x, y, n_in, n_out, s); break;
        case 7: launch_cols<7>(qs, d, x, y, n_in, n_out, s); break;
        default: launch_cols<8>(qs, d, x, y, n_in, n_out, s); break;
    }
    return true;
}

// GGUF Q8_0 blocks (34 bytes, 2-byte aligned) -> the packed planes, on the device (the head's 0.6 GiB stays off the host)
__global__ void pack_kernel(const Q80Block* __restrict__ blocks, int8_t* __restrict__ qs, half* __restrict__ d,
                            std::size_t count) {
    const std::size_t b = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= count) return;
    const Q80Block& src = blocks[b];
    const uint16_t* in = reinterpret_cast<const uint16_t*>(src.qs);
    uint16_t* out = reinterpret_cast<uint16_t*>(qs + b * 32);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = in[i];
    d[b] = src.d;
}

} // namespace q8sm60

// p100 (opt-in STRATA_Q8_SM60_GGUF=1, with STRATA_Q8_SM60=1): the same GEMV read straight from the GGUF blocks (34 bytes
// per 32 values, 2-byte aligned), so the packed planes above - a second copy of every dense matrix and the head, 2.9 GiB
// over Flash-Next's two stages - are not needed.  A warp owns a row (rows of 2048+ inputs, as `kernel` with TPR 32) and
// stages it through shared memory in segments of SEG blocks with aligned 16-byte loads (a row is n_in / 32 * 34 bytes,
// a multiple of 16 when n_in % 256 == 0), the next segment's loads in flight while the current one is used: the
// llama.cpp P100 fork's staging of blocks that are 2 mod 4 bytes (Keoian/dual-p100-qwen-3-8-27b, 4d9dbeb34 /
// be811a6d1).  Lane chunk c (16 values, half of block c / 2) runs in ascending order with `kernel`'s dp4a chain, float
// expression and shuffle sum, so every output is bitwise `kernel`'s (checked on every shape and column count).
namespace q8sm60g {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int SEG = 64;                   // GGUF blocks per staged segment: 2176 bytes
constexpr int SEG16 = SEG * 34 / 16;      // 136 int4
constexpr int PF = (SEG16 + 31) / 32;     // int4 per lane per segment
constexpr int MAX_COLS = 4;               // columns of one launch (more run in passes: each column is its own sum)

inline std::size_t smem_bytes(int ncols, int n_in) {
    return std::size_t(ncols) * n_in + std::size_t(ncols) * (n_in / 32) * sizeof(float) + std::size_t(WARPS) * SEG16 * 16;
}

// 4 bytes at a 2-byte aligned shared address
__device__ __forceinline__ int ld_u32_a2(const uint8_t* p) {
    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0) return *reinterpret_cast<const int*>(p);
    const unsigned lo = *reinterpret_cast<const unsigned*>(p - 2), hi = *reinterpret_cast<const unsigned*>(p + 2);
    return (int) __funnelshift_r(lo, hi, 16);
}

template<int NCOLS>
__launch_bounds__(THREADS)
__global__ void kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y, int n_in,
                       int n_out) {
    extern __shared__ int4 q8sm60g_smem[];
    const int nb = n_in / 32;
    const std::size_t row_bytes = std::size_t(nb) * 34;
    int8_t* sx = reinterpret_cast<int8_t*>(q8sm60g_smem);                       // [NCOLS][n_in]
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * n_in);       // [NCOLS][nb]
    int4* swall = reinterpret_cast<int4*>(sd + std::size_t(NCOLS) * nb);        // [WARPS][SEG16]
    for (int b = int(threadIdx.x); b < NCOLS * nb; b += THREADS) {               // as `kernel`
        const Q81Block* xb = x + b;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = src[i];
        sd[b] = __low2float(xb->ds);
    }
    __syncthreads();
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) >> 5;
    int4* sw = swall + warp * SEG16;
    const uint8_t* swb = reinterpret_cast<const uint8_t*>(sw);
    const int nseg = (nb + SEG - 1) / SEG;
    const int n16_row = int(row_bytes / 16);
    // the warp's work as a sequence of (row, segment) items; item k + 1's loads are in flight while item k is used,
    // across row boundaries too (a row's first segment no longer waits on DRAM after the previous row's sum)
    const int row0 = int(blockIdx.x) * WARPS + warp, rstride = int(gridDim.x) * WARPS;
    auto fetch = [&](int row, int sg, int4 (&pf)[PF]) {
        const int4* src = reinterpret_cast<const int4*>(w + std::size_t(row) * row_bytes);
#pragma unroll
        for (int k = 0; k < PF; ++k) {
            const int i = sg * SEG16 + lane + 32 * k;
            pf[k] = (lane + 32 * k < SEG16 && i < n16_row) ? __ldg(src + i) : make_int4(0, 0, 0, 0);
        }
    };
    int4 pf[PF];
    if (row0 < n_out) fetch(row0, 0, pf);
    float acc[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
    for (int row = row0, sg = 0; row < n_out;) {
        __syncwarp();                                                             // the previous segment is consumed
#pragma unroll
        for (int k = 0; k < PF; ++k)
            if (lane + 32 * k < SEG16) sw[lane + 32 * k] = pf[k];
        __syncwarp();
        int nrow = row, nsg = sg + 1;                                             // the next item
        if (nsg == nseg) { nsg = 0; nrow = row + rstride; }
        if (nrow < n_out) fetch(nrow, nsg, pf);
        const int b0 = sg * SEG, nbs = min(SEG, nb - b0);
        for (int cl = lane; cl < 2 * nbs; cl += 32) {                            // c = 2 * b0 + cl, ascending per lane
            const int c = 2 * b0 + cl;
            const uint8_t* blk = swb + std::size_t(cl >> 1) * 34;
            const float dw = __half2float(*reinterpret_cast<const half*>(blk));
            const uint8_t* q = blk + 2 + (cl & 1) * 16;
            const int w0 = ld_u32_a2(q), w1 = ld_u32_a2(q + 4), w2 = ld_u32_a2(q + 8), w3 = ld_u32_a2(q + 12);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * n_in)[c];
                int s = STRATA_DP4A(w0, xv.x, 0);
                s = STRATA_DP4A(w1, xv.y, s);
                s = STRATA_DP4A(w2, xv.z, s);
                s = STRATA_DP4A(w3, xv.w, s);
                acc[j] += dw * sd[j * nb + (c >> 1)] * float(s);
            }
        }
        if (nsg == 0) {                                                           // the row is done
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                float v = acc[j];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
                if (lane == 0) y[std::size_t(j) * n_out + row] = v;
                acc[j] = 0.0f;
            }
        }
        row = nrow;
        sg = nsg;
    }
}

// the shapes it reads from the GGUF blocks: rows of 2048+ inputs (a warp a row, as `kernel`), whole 16-byte rows,
// a 16-byte aligned matrix, MAX_COLS columns' planes in shared memory
inline bool eligible(const void* w, int n_in, int n_out) {
    return w != nullptr && n_out > 0 && n_in >= 2048 && n_in % 256 == 0 && reinterpret_cast<std::uintptr_t>(w) % 16 == 0 &&
           smem_bytes(MAX_COLS, n_in) <= q8sm60::MAX_SMEM;
}

template<int NCOLS>
void launch_cols(const uint8_t* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int groups = (n_out + WARPS - 1) / WARPS;
    const int grid = (std::min)(groups, q8sm60::sm_count() * q8sm60::blocks_per_sm());
    kernel<NCOLS><<<unsigned(grid), THREADS, smem_bytes(NCOLS, n_in), s>>>(w, x, y, n_in, n_out);
}

// ncols any: passes of MAX_COLS columns (each column's sum is its own, so the passes are bitwise one launch)
inline bool launch(const void* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (!eligible(w, n_in, n_out) || ncols < 1) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    for (int j = 0; j < ncols; j += MAX_COLS) {
        const int nj = (std::min)(MAX_COLS, ncols - j);
        const Q81Block* xj = x + std::size_t(j) * std::size_t(n_in / 32);
        float* yj = y + std::size_t(j) * std::size_t(n_out);
        switch (nj) {
            case 1: launch_cols<1>(wb, xj, yj, n_in, n_out, s); break;
            case 2: launch_cols<2>(wb, xj, yj, n_in, n_out, s); break;
            case 3: launch_cols<3>(wb, xj, yj, n_in, n_out, s); break;
            default: launch_cols<4>(wb, xj, yj, n_in, n_out, s); break;
        }
    }
    return true;
}

} // namespace q8sm60g

// p100 (STRATA_Q8_SM60_GGUF=2): the GGUF-block GEMV with aligned loads and no weight staging.  The window's activations
// are laid out in shared memory as the weights are - 34-byte blocks with zeros where a block's fp16 scale sits - so a
// lane reads 16 bytes of a row straight from DRAM and dots them with the 16 matching activation bytes.  A 4-byte word of
// a GGUF row holds the qs of one block only (block floor(offset / 34)); its scale bytes meet the zeroed activations.  The
// int8 dots are exact; the float sums are taken per (block, 16-byte piece) instead of per half block, so the outputs
// match q8sm60::kernel to rounding, not bitwise (max relative error at the packed kernel's level).  Measured on 2x
// P100: 25-60% slower than q8sm60g (scattered 2-byte scale loads), decode -3%: not used.
namespace q8sm60g2 {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAX_COLS = 4;

inline std::size_t smem_bytes(int ncols, int n_in) {
    const std::size_t nb = std::size_t(n_in / 32);
    return std::size_t(ncols) * nb * 34 + std::size_t(ncols) * nb * sizeof(float) + 16;
}

template<int NCOLS>
__launch_bounds__(THREADS)
__global__ void kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y, int n_in,
                       int n_out) {
    extern __shared__ int4 q8sm60g2_smem[];
    const int nb = n_in / 32;
    const int row_bytes = nb * 34, n16 = row_bytes / 16;                         // n_in % 256 == 0: whole int4 a row
    uint8_t* sx = reinterpret_cast<uint8_t*>(q8sm60g2_smem);                    // [NCOLS][row_bytes], d positions 0
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * row_bytes);  // [NCOLS][nb]
    for (int b = int(threadIdx.x); b < NCOLS * nb; b += THREADS) {
        const int j = b / nb, bb = b - j * nb;
        uint8_t* dst = sx + std::size_t(j) * row_bytes + std::size_t(bb) * 34;
        *reinterpret_cast<uint16_t*>(dst) = 0;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(x[b].qs);
#pragma unroll
        for (int i = 0; i < 16; ++i) reinterpret_cast<uint16_t*>(dst + 2)[i] = src[i];
        sd[b] = __low2float(x[b].ds);
    }
    __syncthreads();
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) >> 5;
    for (int row = int(blockIdx.x) * WARPS + warp; row < n_out; row += int(gridDim.x) * WARPS) {
        const uint8_t* rw = w + std::size_t(row) * row_bytes;
        const int4* r4 = reinterpret_cast<const int4*>(rw);
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        for (int i = lane; i < n16; i += 32) {
            const int4 wv = __ldg(r4 + i);
            const int o = 16 * i;
            const int b0 = o / 34, b1 = (o + 12) / 34;                            // the blocks of the first and last word
            const int split = b1 != b0 ? (34 * b1 - o + 3) / 4 : 4;               // words k >= split belong to b1
            const float dw0 = __half2float(__ldg(reinterpret_cast<const __half*>(rw + 34 * b0)));
            const float dw1 = b1 != b0 ? __half2float(__ldg(reinterpret_cast<const __half*>(rw + 34 * b1))) : 0.0f;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * row_bytes)[i];
                const int p0 = STRATA_DP4A(wv.x, xv.x, 0), p1 = STRATA_DP4A(wv.y, xv.y, 0);
                const int p2 = STRATA_DP4A(wv.z, xv.z, 0), p3 = STRATA_DP4A(wv.w, xv.w, 0);
                int s0, s1;
                if (split == 4) { s0 = p0 + p1 + p2 + p3; s1 = 0; }
                else if (split == 1) { s0 = p0; s1 = p1 + p2 + p3; }
                else if (split == 2) { s0 = p0 + p1; s1 = p2 + p3; }
                else { s0 = p0 + p1 + p2; s1 = p3; }
                acc[j] += dw0 * sd[j * nb + b0] * float(s0);
                if (split != 4) acc[j] += dw1 * sd[j * nb + b1] * float(s1);
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
            if (lane == 0) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

inline bool eligible(const void* w, int n_in, int n_out) {
    return w != nullptr && n_out > 0 && n_in >= 2048 && n_in % 256 == 0 && reinterpret_cast<std::uintptr_t>(w) % 16 == 0 &&
           smem_bytes(MAX_COLS, n_in) <= q8sm60::MAX_SMEM;
}

template<int NCOLS>
void launch_cols(const uint8_t* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int groups = (n_out + WARPS - 1) / WARPS;
    const int grid = (std::min)(groups, q8sm60::sm_count() * q8sm60::blocks_per_sm());
    kernel<NCOLS><<<unsigned(grid), THREADS, smem_bytes(NCOLS, n_in), s>>>(w, x, y, n_in, n_out);
}

inline bool launch(const void* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (!eligible(w, n_in, n_out) || ncols < 1) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    for (int j = 0; j < ncols; j += MAX_COLS) {
        const int nj = (std::min)(MAX_COLS, ncols - j);
        const Q81Block* xj = x + std::size_t(j) * std::size_t(n_in / 32);
        float* yj = y + std::size_t(j) * std::size_t(n_out);
        switch (nj) {
            case 1: launch_cols<1>(wb, xj, yj, n_in, n_out, s); break;
            case 2: launch_cols<2>(wb, xj, yj, n_in, n_out, s); break;
            case 3: launch_cols<3>(wb, xj, yj, n_in, n_out, s); break;
            default: launch_cols<4>(wb, xj, yj, n_in, n_out, s); break;
        }
    }
    return true;
}

} // namespace q8sm60g2
