#include "losses.h"

Loss::Loss(float value, float_batch delta)
{
    this->value = value;
    this->delta = delta;
}

Loss MSELoss::apply(float_batch pred, float_batch target)
{
    this->pred = pred;
    this->target = target;
    int w = pred.size(), h = pred[0].size();
    float loss = 0;
    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            loss += pow(pred[i][j] - target[i][j], 2) / 2;
        }
    }
    return Loss(loss / w, this->delta());
}

float_batch MSELoss::delta()
{
    float_batch delta = this->utils.mat_add(this->pred, this->utils.rescale(this->target, -1));
    return delta;
}

Loss CrossEntropyLoss::apply(float_batch pred, float_batch target)
{
    this->pred = pred;
    this->target = target;
    int w = pred.size();
    float_batch probs = this->soft_max(pred);
    float loss = 0;
    for (int i = 0; i < w; i++){
        loss += -log(probs[i][(int)target[i][0]]);
    }
    return Loss(loss / w, this->delta());
}

float_batch CrossEntropyLoss::delta()
{
    int w = this->pred.size();
    float_batch probs = this->soft_max(pred);
    for (int i = 0; i < w; i++){
        probs[i][(int)target[i][0]] -= 1;
    }
    return probs;
}

float_batch CrossEntropyLoss::soft_max(float_batch x)
{
    int w = x.size(), h = x[0].size();
    float_batch num(w, vector<float>(h, 1));
    vector<float> den(w, 1);
    for (int i = 0; i < w; i++){
        float max_of_batch = -std::numeric_limits<float>::max();
        float sum_of_batch = 0;
        for (int j = 0; j < h; j++){
            if (x[i][j] > max_of_batch){
                max_of_batch = x[i][j];
            }
        }
        for (int j = 0; j < h; j++){
            num[i][j] = exp(x[i][j] - max_of_batch);
            sum_of_batch += num[i][j];
        }
        den[i] = sum_of_batch;
    }
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            num[i][j] = num[i][j] / den[i] + 0.000001;
        }
    }
    return num;
}
