#include "demonn_core.h"
#include <mkl.h>
#include <mkl_vsl.h>
#include <ctime>

namespace demonn_core {

    void normal_distribution(
        float* data, int count,
        float mean, float std
    ) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, (unsigned long long)clock());
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, count, data, mean, std);
        vslDeleteStream(&stream);
    }

}