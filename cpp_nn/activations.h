#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include <vector>
#include <array>

using namespace std;

typedef vector<vector<float>> float_batch;


class Activation
{
public:
    virtual float_batch forward(const float_batch &x)=0;
    virtual float_batch derivative(const float_batch &x)=0;

};

class ReLU : public Activation
{
public:
    float_batch forward(const float_batch &x);
    float_batch derivative(const float_batch &x);
};

class Linear : public Activation
{
public:
    float_batch forward(const float_batch &x);
    float_batch derivative(const float_batch &x);
};


#endif // ACTIVATIONS_H
