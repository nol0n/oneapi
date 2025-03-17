#include <chrono>
#include <device.hpp>
#include <iomanip>
#include <iostream>
#include <sycl/device_selector.hpp>
#include <vector>

#include "gemm_block_oneapi.h"
#include "rng.hpp"

int main() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(3);

  size_t n = 4096;
  
  std::vector<float> a = rng::float_vector(n * n, 0.1f, 0.9f);
  std::vector<float> b = rng::float_vector(n * n, 0.1f, 0.9f);

  size_t n_progrev = 32;

  std::vector<float> a_progrev = rng::float_vector(n_progrev * n_progrev, 0.1f, 0.9f);
  std::vector<float> b_progrev = rng::float_vector(n_progrev * n_progrev, 0.1f, 0.9f);

  sycl::device device = sycl::device(sycl::cpu_selector_v);

  //==================================================================

  std::vector<float> mine = GemmBlockONEAPI(a_progrev, b_progrev, n_progrev, device);
  
  // time
  start = std::chrono::high_resolution_clock::now();
  mine = GemmBlockONEAPI(a, b, n, device);
  end = std::chrono::high_resolution_clock::now();

  seconds = end - start;
  std::cout << " - " << seconds.count() << " s.\n";

  for (int i = 50; i < 55; i++)
    std::cout << mine[i] << " ";
  std::cout << "\n";

  //==================================================================

  return 0;
}