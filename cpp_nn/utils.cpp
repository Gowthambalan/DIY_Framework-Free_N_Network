#include "utils.h"
#include <iostream>

float_batch Utils::mat_mul(const float_batch &A, const float_batch &B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(m != k){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = 0;
            for(unsigned int r = 0; r < k; r++){
                temp[i][j] += A[i][r] * B[r][j];
            }
        }
    }
    return temp;
}

float_batch Utils::element_wise_mul(const float_batch &A, const float_batch &B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] * B[i][j];
        }
    }
    return temp;
}

float_batch Utils::mat_add(const float_batch &A, const float_batch &B)
{
    unsigned int n = A.size(), k = B.size();
    unsigned int m = A[0].size(), l = B[0].size();

    if(n != k || m != l){
        throw "Invalid matrices' shapes!";
    }
    float_batch temp(n, vector<float>(l, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < l; j++){
            temp[i][j] = A[i][j] + B[i][j];
        }
    }
    return temp;
}


float_batch Utils::rescale(const float_batch &A, float scale)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = A[i][j] * scale;
        }
    }
    return temp;
}

float_batch Utils::add_scalar(const float_batch &A, float scalar)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = A[i][j] + scalar;
        }
    }
    return temp;
}

float_batch Utils::transpose(const float_batch &A)
{
    unsigned int n = A.size(), m = A[0].size();

    float_batch temp(m, vector<float>(n, 1));
    for(unsigned int i = 0; i < m; i++){
        for(unsigned int j = 0; j < n; j++){
            temp[i][j] = A[j][i];
        }
    }
    return temp;

}

float_batch Utils::element_wise_sqrt(const float_batch &A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] = sqrt(A[i][j]);
        }
    }
    return temp;

}

float_batch Utils::element_wise_rev(const float_batch &A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(n, vector<float>(m, 1));
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < m; j++){
            temp[i][j] =  1 / A[i][j];
        }
    }
    return temp;

}

float_batch Utils::batch_mean(const float_batch &A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(1, vector<float>(m, 1));
    for (vector<float> a : A) {
        for (unsigned int j = 0; j < m; j++) {
            temp[0][j] += (a[j] / n);
        }
    }
    return temp;
}

float_batch Utils::batch_var(const float_batch &A, const float_batch &mu)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(1, vector<float>(m, 1));
    for (vector<float> a : A) {
        for (unsigned int j = 0; j < m; j++) {
            temp[0][j] += (pow(a[j] - mu[0][j], 2) / n);
        }
    }
    return temp;
}

float_batch Utils::batch_sum(const float_batch &A)
{
    unsigned int n = A.size(), m = A[0].size();
    float_batch temp(1, vector<float>(m, 1));
    for (vector<float> a : A) {
        for (unsigned int j = 0; j < m; j++) {
            temp[0][j] += a[j];
        }
    }
    return temp;
}

vector<float_batch> Utils::equal_batch_size(const float_batch &A, const float_batch &B)
{
    size_t w0 = A.size(), w1 = B.size();
    size_t w;
    float_batch X, temp;
    if (w0 < w1){
        w = w1;
        temp = float_batch(w1, vector<float>(A[0].size(), 1));
        X = A;
    }
    else {
        w = w0;
        temp = float_batch(w0, vector<float>(B[0].size(), 1));
        X = B;
    }
    for(size_t i = 0; i < w; i++){
        temp[i] = X[0];
    }
    if (w0 < w1) {
        return vector<float_batch>{temp, B};
    }
    else{
        return vector<float_batch>{A, temp};
    }
}
