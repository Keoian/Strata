// src/kernels/cpu/q2_avx1_parity.cpp - a fingerprint of the AVX1 Q2_0 rows (q2_avx1.cpp) on fixed random blocks, for
// checking that a change to those kernels keeps their bytes: build it before and after, run both with the same
// STRATA_Q2_BITPLANE, compare the printed hashes.  No GPU, no model.  Prints one FNV-1a hash per token count (1..8)
// over the outputs of a gate/up-shaped matrix (640 rows of 2560) and a down-shaped one (2560 rows of 640).
//
//     STRATA_Q2_BITPLANE=1 q2_avx1_parity
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

uint16_t f2h(float f) {   // fp32 -> fp16, round to nearest even, normal range only (the scales here)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t) sign;
    uint32_t h = sign | ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// rows x (cols / 64) Q2_0 blocks: an fp16 scale and 16 bytes of 2-bit codes each
std::vector<uint8_t> make_rows(int rows, int cols, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) rows * (cols / 64) * 18);
    std::uniform_real_distribution<float> sd(0.002f, 0.03f);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < w.size() / 18; ++b) {
        const uint16_t d = f2h(sd(rng));
        std::memcpy(&w[b * 18], &d, 2);
        for (int i = 0; i < 16; ++i) w[b * 18 + 2 + i] = (uint8_t) byte(rng);
    }
    return w;
}

uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

}  // namespace

int main() {
    if (!cpu::cpu_avx1_ok()) { std::printf("SKIP: no AVX\n"); return 0; }
    constexpr int H = 2560, FF = 640, NTMAX = 8;
    std::mt19937 rng(20261030);
    const std::vector<uint8_t> gu = make_rows(FF, H, rng), dn = make_rows(H, FF, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto ax = std::make_unique<cpu::ActQ[]>(NTMAX), ah = std::make_unique<cpu::ActQ[]>(NTMAX);
    for (int t = 0; t < NTMAX; ++t) {
        std::vector<float> x(H), h(FF);
        for (auto& v : x) v = nd(rng);
        for (auto& v : h) v = 0.1f * nd(rng);
        cpu::act_quant_any(x.data(), H, ax[t]);
        cpu::act_quant_any(h.data(), FF, ah[t]);
    }
    std::vector<float> og((size_t) NTMAX * FF), od((size_t) NTMAX * H);
    for (int nt = 1; nt <= NTMAX; ++nt) {
        const cpu::ActQ* pa[NTMAX];
        const cpu::ActQ* ph[NTMAX];
        float* pg[NTMAX];
        float* pd[NTMAX];
        for (int t = 0; t < NTMAX; ++t) { pa[t] = &ax[t]; ph[t] = &ah[t]; pg[t] = &og[(size_t) t * FF]; pd[t] = &od[(size_t) t * H]; }
        cpu::q2_rows_any(gu.data(), (H / 64) * 18, H / 64, pa, nt, pg, 0, FF);
        cpu::q2_rows_any(dn.data(), (FF / 64) * 18, FF / 64, ph, nt, pd, 0, H);
        uint64_t hsh = 1469598103934665603ull;
        hsh = fnv(og.data(), (size_t) nt * FF * sizeof(float), hsh);
        hsh = fnv(od.data(), (size_t) nt * H * sizeof(float), hsh);
        std::printf("nt %d: %016llx  (gate[0] %.9g, down[0] %.9g)\n", nt, (unsigned long long) hsh, og[0], od[0]);
    }
    return 0;
}
