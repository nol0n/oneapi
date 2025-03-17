#include "gemm_mkl_oneapi.h"
#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <usm.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(const std::vector<float> a,
                                 const std::vector<float> b, size_t size,
                                 sycl::device device) {
  std::vector<float> ans(size * size, 0.0f);
  return ans;
}
