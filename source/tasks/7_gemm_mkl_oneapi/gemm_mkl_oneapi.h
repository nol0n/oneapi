#ifndef __GEMM_MKL_ONEAPI_H
#define __GEMM_MKL_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device);

#endif  // __GEMM_MKL_ONEAPI_H