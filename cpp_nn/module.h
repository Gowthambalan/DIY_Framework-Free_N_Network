#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "layers.h"
#include <losses.h>

using namespace std;


class Module
{
public:
    vector<Layer*> parameters;
    virtual  float_batch forward(const float_batch &input, bool eval)=0;
    void backward(const Loss &loss);
};

#endif // MODULE_H
