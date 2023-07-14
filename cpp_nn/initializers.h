#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <random>
#include <vector>
#include <stdexcept>

using namespace std;

typedef vector<vector<float>> float_batch;


class Initializer{
public:
    virtual float_batch initialize(unsigned int w, unsigned int h)=0;
};

class Constant : public Initializer{
public:
    Constant(float c){
        this->c = c;
    }
    float_batch initialize(unsigned int w, unsigned int h);

private:
    float c = 0;
};

class RandomUniform : public Initializer{
public:
    float_batch initialize(unsigned int w, unsigned int h);

private:
    std::random_device rd;
    std::uniform_real_distribution<float> dis{0.0, 1.0};
};

class XavierUniform : public Initializer{
public:
    float_batch initialize(unsigned int fan_in, unsigned int fan_out);

private:
    std::random_device rd;
};

class HeNormal : public Initializer{
public:
    float_batch initialize(unsigned int fan_in, unsigned int fan_out);
    HeNormal(string non_linearity, string mode);

private:
    std::random_device rd;
    string non_linearity;
    string mode;


};
#endif // INITIALIZERS_H
