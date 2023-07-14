#ifndef LAYERS_H
#define LAYERS_H

#include <initializers.h>
#include <utils.h>
#include <activations.h>
#include <array>

using namespace std;

class Layer{

public:
    float_batch W, dW, b, db;
    float lambda = 0;

    virtual float_batch forward(const float_batch &x, bool eval) = 0;
    virtual float_batch backward(float_batch &x) = 0;
};


class Dense : public Layer{

public:
    int in_features, out_features;
    string weight_initializer;
    string bias_initializer;
    string activation = "linear";
    string regularization_type;
    float_batch input, z;

    RandomUniform random_uniform;
    Constant zeros{0.0};
    XavierUniform xavier_uniform;
    Linear linear;
    ReLU relu;

    Dense(int in_features, int out_features,
          string activation,
          string weight_initializer,
          string bias_initialzer,
          string regularization_type,
          float lambda
          ){
        this->in_features = in_features;
        this->out_features = out_features;
        this->weight_initializer = weight_initializer;
        this->bias_initializer = bias_initialzer;
        this->activation = activation;
        this->regularization_type = regularization_type;
        this->lambda = lambda;

        HeNormal he_normal(this->activation, "fan_in");

        if (!weight_initializer.compare("xavier_uniform")){
            this->W = this->xavier_uniform.initialize(this->in_features, this->out_features);
        }
        else if (!weight_initializer.compare("he_normal")){
            this->W = he_normal.initialize(this->in_features, this->out_features);
        }
        else{
            this->W = this->random_uniform.initialize(this->in_features, this->out_features);
        }
        if (!bias_initializer.compare("zeros")) {
            this->b = this->zeros.initialize(1, this->out_features);
        }

        this->dW = this->zeros.initialize(this->in_features, this->out_features);
        this->db = this->zeros.initialize(1, this->out_features);
    };



    // Layer interface
    float_batch forward(const float_batch &x, bool eval);
    float_batch backward(float_batch &delta);
};


class BatchNorm1d : public Layer{

public:
    int in_features;
    string weight_initializer;
    string bias_initializer;
    float_batch mu_hat, std_hat;
    float_batch mu, std;
    float_batch x_hat, gamma;
    float beta = 1, eps = 0.00001;

    Constant zeros{0.0};
    Constant ones{1.0};


    BatchNorm1d(int in_features){
        this->in_features = in_features;

        this->W = this->ones.initialize(1, this->in_features);
        this->b = this->zeros.initialize(1, this->in_features);
        this->std_hat = this->ones.initialize(1, this->in_features);
        this->mu_hat = this->zeros.initialize(1, this->in_features);

    };



    // Layer interface
    float_batch forward(const float_batch &x, bool eval);
    float_batch backward(float_batch &delta);
};
#endif // LAYERS_H
