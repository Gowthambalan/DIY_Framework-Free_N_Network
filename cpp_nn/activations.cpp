#include "activations.h"


float_batch Linear::forward(const float_batch &x)
{
    return x;

}

float_batch Linear::derivative(const float_batch &x)
{
    int w = x.size(), h = x[0].size();
    float_batch temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = 1;
        }
    }
    return temp;
}

float_batch ReLU::forward(const float_batch &x)
{
    int w = x.size(), h = x[0].size();
    float_batch temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = (x[i][j] > 0) ? x[i][j] : 0;
        }
    }
    return temp;
}

float_batch ReLU::derivative(const float_batch &x)
{
    int w = x.size(), h = x[0].size();
    float_batch temp(w, vector<float>(h, 1));

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            temp[i][j] = (x[i][j] > 0) ? 1 : 0;
        }
    }
    return temp;
}
