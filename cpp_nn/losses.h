#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <math.h>
#include <utils.h>
#include <limits>

using namespace std;

typedef vector<vector<float>> float_batch;


class Loss
{
public:
    float value;
    vector<vector<float>> delta;
    Loss(float value, float_batch delta);
};

class LossFunc
{
public:
    float_batch target, pred;
    virtual Loss apply(float_batch pred, float_batch target)=0;
    virtual float_batch delta()=0;
};

class MSELoss : LossFunc
{
public:
    Utils utils;
    Loss apply(float_batch pred, float_batch target);
    float_batch delta();
};

class CrossEntropyLoss : LossFunc
{
public:
    Utils utils;
    Loss apply(float_batch pred, float_batch target);
    float_batch delta();
    float_batch soft_max(float_batch x);
};
#endif // LOSS_H
