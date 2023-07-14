#include "layers.h"


float_batch Dense::forward(const float_batch &x, bool eval)
{
    this->input = x;
    float_batch z = Utils::mat_mul(x, this->W);
    int x_dim0 = x.size();
    float_batch b(x_dim0, vector<float>(this->b[0].size()));
    for(int i = 0; i < x_dim0; i++){
        b[i] = this->b[0];
    }
    z = Utils::mat_add(z, b);
    this->z = z;
    float_batch a = (!this->activation.compare("relu")) ? this->relu.forward(z) : this->linear.forward(z);

    return a;
}

float_batch Dense::backward(float_batch &delta)
{
    float_batch dz;
    if(!this->activation.compare("relu")){
        dz = Utils::element_wise_mul(delta, this->relu.derivative(this->z));
    }
    else{
        dz = Utils::element_wise_mul(delta, this->linear.derivative(this->z));
    }
    float_batch input_t = Utils::transpose(this->input);
    float_batch dw = Utils::mat_mul(input_t, dz);
    this->dW = Utils::rescale(dw, 1.0 / dz.size());

    if (!this->regularization_type.compare("l2")){
        this->dW = Utils::mat_add(this->dW, Utils::rescale(this->W, this->lambda));
    }
    else if(!this->regularization_type.compare("l1")){
        this->dW = Utils::add_scalar(this->dW, this->lambda);
    }

    float_batch ones_t(1, vector<float>(dz.size(), 1));
    for(size_t i = 0; i < ones_t.size(); i++){
        for(size_t j = 0; j < ones_t[0].size(); j++){
            ones_t[i][j] = 1;
        }
    }
    float_batch db = Utils::mat_mul(ones_t, dz);
    this->db = Utils::rescale(db, 1.0 / dz.size());


    float_batch w_t = Utils::transpose(this->W);
    delta = Utils::mat_mul(dz, w_t);
    return delta;
}

float_batch BatchNorm1d::forward(const float_batch &x, bool eval)
{
    if (!eval){
        this->mu = Utils::batch_mean(x);
        this->std = Utils::element_wise_sqrt(Utils::batch_var(x, this->mu));
        this->mu_hat = Utils::mat_add(Utils::rescale(this->mu_hat, 1 - this->beta), Utils::rescale(this->mu, this->beta));
        this->std_hat = Utils::mat_add(Utils::rescale(this->std_hat, 1 - this->beta), Utils::rescale(this->std, this->beta));
    }
    else {
        this->mu = this->mu_hat;
        this->std = this->std_hat;
    }
    float_batch mu(x.size(), vector<float>(this->mu[0].size(), 1));
    float_batch std(x.size(), vector<float>(this->std[0].size(), 1));
    for (size_t i = 0; i < x.size(); i++){
        mu[i] = this->mu[0];
        std[i] = this->std[0];
    }
    float_batch num = Utils::mat_add(x, Utils::rescale(mu, -1));
    float_batch den = Utils::element_wise_sqrt(Utils::add_scalar(Utils::element_wise_mul(std, std), this->eps));
    float_batch x_hat = Utils::element_wise_mul(num, Utils::element_wise_rev(den));
    this->x_hat = x_hat;

    this->gamma = float_batch(x.size(), vector<float>(this->W[0].size(), 1));
    float_batch beta = float_batch(x.size(), vector<float>(this->b[0].size(), 1));
    for (size_t i = 0; i < x.size(); i++){
        this->gamma[i] = this->W[0];
        beta[i] = this->b[0];
    }
    float_batch y = Utils::mat_add(Utils::element_wise_mul(this->gamma, x_hat), beta);
    return y;

}

float_batch BatchNorm1d::backward(float_batch &delta)
{
    float_batch dz = delta;
    float_batch dx_hat = Utils::element_wise_mul(dz, this->gamma);
    int m = dz.size();
    this->dW = Utils::rescale(Utils::batch_sum(Utils::element_wise_mul(this->x_hat, dz)), 1 / m);
    this->db = Utils::rescale(Utils::batch_sum(dz), 1 / m);

    float_batch a1 = Utils::rescale(dx_hat, m);
    float_batch a2 = Utils::batch_sum(dx_hat);
    vector<float_batch> temp = Utils::equal_batch_size(this->x_hat, Utils::batch_sum(Utils::element_wise_mul(dx_hat, this->x_hat)));
    float_batch a3 = Utils::element_wise_mul(temp[0], temp[1]);
    temp = Utils::equal_batch_size(Utils::rescale(a2, -1), Utils::rescale(a3, -1));
    float_batch num = Utils::mat_add(a1, Utils::mat_add(temp[0], temp[1]));
    float_batch den = Utils::rescale(Utils::element_wise_sqrt(Utils::add_scalar(Utils::element_wise_mul(this->std, this->std), this->eps)), m);

    temp = Utils::equal_batch_size(num , Utils::element_wise_rev(den));
    delta = Utils::element_wise_mul(temp[0], temp[1]);
    return delta;
}
