#include "initializers.h"


float_batch Constant::initialize(unsigned int w, unsigned int h)
{
    float_batch temp(w, vector<float>(h, 1));

    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->c;
        }
    }
    return temp;
}

float_batch RandomUniform::initialize(unsigned int w, unsigned int h){
    float_batch temp(w, vector<float>(h, 1));
    std::mt19937 gen(this->rd());
    for(unsigned int i  = 0; i < w; i++){
        for(unsigned int j = 0; j < h; j++){
            temp[i][j] = this->dis(gen);
        }
    }
    return temp;
}

float_batch XavierUniform::initialize(unsigned int fan_in, unsigned int fan_out)
{
    float std = sqrt(2 / (fan_in + fan_out));
    float a = std * sqrt(3);

    float_batch temp(fan_in, vector<float>(fan_out, 1));
    std::mt19937 gen(this->rd());
    std::uniform_real_distribution<float> dis{-a, a};
    for(unsigned int i  = 0; i < fan_in; i++){
        for(unsigned int j = 0; j < fan_out; j++){
            temp[i][j] = dis(gen);
        }
    }
    return temp;

}

HeNormal::HeNormal(string non_linearity, string mode)
{
    this->non_linearity = non_linearity;
    this->mode = mode;
    if (this->mode.compare("fan_in") && this->mode.compare("fan_out")){
        throw std::invalid_argument("Invalid mode!");
    }
}

float_batch HeNormal::initialize(unsigned int fan_in, unsigned int fan_out)
{
    int fan = !this->mode.compare("fan_in") ? fan_in : fan_out;
    float gain;
    if (!this->non_linearity.compare("relu")){
        gain = sqrt(2);
    }
    else{
        throw std::runtime_error("invalid nonlinearity");
    }
    float std = gain / sqrt(fan);

    float_batch temp(fan_in, vector<float>(fan_out, 1));
    std::mt19937 gen(this->rd());
    std::normal_distribution<float> dis{0, std};
    for(unsigned int i  = 0; i < fan_in; i++){
        for(unsigned int j = 0; j < fan_out; j++){
            temp[i][j] = dis(gen);
        }
    }
    return temp;
}

