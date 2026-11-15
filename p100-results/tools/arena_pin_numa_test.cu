#include <sys/mman.h>
#include <sys/syscall.h>
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <linux/io_uring.h>
#include <sys/uio.h>
#include <cstring>

static uint64_t blob = 2764800, n;
static uint8_t* base;
static std::vector<std::pair<uint64_t,uint64_t>> r[2];
// FOLL_LONGTERM pins from THIS thread (io_uring fixed buffers): CMA-block pages migrate now, to this thread's node
static long pin_ranges(const std::vector<std::pair<uint64_t,uint64_t>>& rs, size_t lo, size_t hi) {
  io_uring_params p; memset(&p, 0, sizeof p);
  int fd = (int) syscall(SYS_io_uring_setup, 1, &p);
  if (fd < 0) return -1;
  std::vector<iovec> iv;
  long n = 0;
  for (size_t i = lo; i < hi; ++i) {
    iv.push_back({base + rs[i].first, (size_t) (rs[i].second - rs[i].first)});
    if (iv.size() == 16384 || i + 1 == hi) {
      if (syscall(SYS_io_uring_register, fd, IORING_REGISTER_BUFFERS, iv.data(), (unsigned) iv.size()) != 0) { close(fd); return -2; }
      syscall(SYS_io_uring_register, fd, IORING_UNREGISTER_BUFFERS, nullptr, 0);
      n += iv.size(); iv.clear();
    }
  }
  close(fd);
  return n;
}
static void where(const char* tag) {
  for (int k = 0; k < 2; ++k) {
    std::vector<void*> pg; for (size_t i = 0; i < r[k].size(); i += 7) pg.push_back(base + r[k][i].first + 4096);
    std::vector<int> st(pg.size(), -9);
    syscall(SYS_move_pages, 0, pg.size(), pg.data(), nullptr, st.data(), 0);
    size_t on = 0; for (int v : st) on += v == k;
    printf("%s slice %d: %zu/%zu on node %d\n", tag, k, on, pg.size(), k);
  }
}
int main(int argc, char** argv) {
  n = argc > 1 ? atoll(argv[1]) : 4000;
  uint64_t bytes = blob * n;
  base = (uint8_t*) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  madvise(base, bytes, MADV_NOHUGEPAGE);
  const uint64_t rb[3] = {921600, 921600, 921600};
  for (uint64_t e = 0; e < n; ++e) { uint64_t s = e * blob; for (int q = 0; q < 3; ++q) { uint64_t m = s + rb[q]/2, en = s + rb[q];
    r[0].push_back({s/4096*4096, (m+4095)/4096*4096}); uint64_t lo=(m+4095)/4096*4096, hi=en/4096*4096; if (hi>lo) r[1].push_back({lo,hi}); s = en; } }
  cudaSetDevice(0); cudaFree(0);
  auto t = [&](int k) { cpu_set_t c; CPU_ZERO(&c); for (int i = 0; i < 72; ++i) if ((k == 0) == ((i % 36) < 18)) CPU_SET(i, &c);
    sched_setaffinity(0, sizeof c, &c); unsigned long m[4] = {1ul << k}; syscall(SYS_set_mempolicy, 2, m, 256);
    const bool pin = getenv("PIN") != nullptr; const size_t step = 2048;
    for (size_t c = 0; c < r[k].size(); c += step) {
      const size_t e = std::min(r[k].size(), c + step);
      for (size_t i = c; i < e; ++i) for (uint64_t a = r[k][i].first; a < r[k][i].second; a += 4096) ((volatile uint8_t*) base)[a] = 0;
      if (pin) { long q = pin_ranges(r[k], c, e); if (q < 0) { printf("pin failed %ld\n", q); exit(1); } }
    } };
  std::thread a(t, 0), b(t, 1); a.join(); b.join();
  where("after touch   ");
  cudaError_t e = cudaHostRegister(base, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
  printf("register: %s\n", cudaGetErrorString(e));
  where("after register");
  for (uint64_t i = 0; i < bytes; i += 4096) base[i] = 1;   // write everything from the main thread, like the loader
  where("after writes  ");
  cudaHostUnregister(base);
  return 0;
}
