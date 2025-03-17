#ifndef RNG_CLASS
#define RNG_CLASS

#include <vector>

namespace rng {
    std::vector<float> float_vector(const std::size_t &size, const float &min, const float &max, const int& t_num = 12);
    std::vector<float> diag_dominant(const std::size_t &size, const float &min, const float &max, const int& t_num = 12);
} // rng namespace

#endif // RNG_CLASS
