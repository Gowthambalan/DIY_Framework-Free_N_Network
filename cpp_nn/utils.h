#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>
#include <functional>
#include <math.h>

using namespace std;

typedef vector<vector<float>> float_batch;

class Utils
{
public:
    static float_batch mat_mul(const float_batch &A, const float_batch &B);
    static float_batch element_wise_mul(const float_batch &A, const float_batch &B);
    static float_batch mat_add(const float_batch &A, const float_batch &B);
    static float_batch rescale(const float_batch &A, float scale);
    static float_batch add_scalar(const float_batch &A, float scalar);
    static float_batch transpose(const float_batch &A);
    static float_batch element_wise_sqrt(const float_batch &A);
    static float_batch element_wise_rev(const float_batch &A);
    static float_batch batch_mean(const float_batch &A);
    static float_batch batch_var(const float_batch &A, const float_batch &mu);
    static float_batch batch_sum(const float_batch &A);
    static vector<float_batch> equal_batch_size(const float_batch &A, const float_batch &B);

};

#endif // UTILS_H
