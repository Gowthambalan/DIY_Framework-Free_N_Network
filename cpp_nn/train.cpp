#include <iostream>
#include <module.h>
#include <layers.h>
#include <utils.h>
#include <losses.h>
#include <optimizers.h>
#include <iomanip>

using namespace std;

class MyNet : public Module{
public:
    int in_features = 0, out_features = 0;
    Dense *hidden, *output;
    BatchNorm1d *bn;

    MyNet(int in_features, int out_features){
        this->in_features = in_features;
        this->out_features = out_features;

        this->hidden = new Dense{this->in_features,
                100,
                "relu",
                "he_normal",
                "zeros",
                "l2",
                0.001};
        this->parameters.push_back(this->hidden);

        this->bn = new BatchNorm1d{100};
        this->parameters.push_back(this->bn);

        this->output = new Dense{100,
                this->out_features,
                "linear",
                "xavier_uniform",
                "zeros",
                "l2",
                0.001};
        this->parameters.push_back(this->output);
    }
    float_batch forward(const float_batch &input, bool eval){
        float_batch x = this->hidden->forward(input, eval);
        x = this->bn->forward(x, eval);
        x = this->output->forward(x, eval);
        return x;

    }
};

void train_regression();
void train_classification();

int main()
{
    train_regression();
    train_classification();
    return 0;
}

void train_classification(){
    cout <<"-----Classification-------"<<endl;
    int num_samples = 100;
    int num_features = 2;
    int num_classes = 3;
    int num_epoch = 1000;
    int batch_size = 64;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 0.2};
    std::uniform_int_distribution<> rand_int(0, num_samples * num_classes - 1);

    float_batch x(num_classes * num_samples, vector<float>(num_features, 1));
    float_batch t(num_classes * num_samples, vector<float>(1, 1));

    float radius[num_samples];
    for (int i = 0; i < num_samples; i++){
        radius[i] = float(i) / num_samples;
    }
    for (int j = 0; j < num_classes; j++){
        float theta[num_samples];
        for(int i = 0; i < num_samples; i++){
            theta[i] = i * 0.04 + j * 4 + d(gen);
        }
        int k = 0;
        for (int idx = j * num_samples; idx < (j + 1) * num_samples; idx++){
            x[idx][0] = radius[k] * sin(theta[k]);
            x[idx][1] = radius[k] * cos(theta[k]);
            t[idx][0] = j;
            k++;
        }
    }

    MyNet my_net = MyNet{num_features, num_classes};
    CrossEntropyLoss celoss;
    Adam opt(0.1, 0.9, 0.999, my_net.parameters);
    float smoothed_loss = 0, total_loss = 0;
    bool smoothed_flag = false;
    float_batch y;
    for(int step = 0; step < num_epoch; step++){
        float_batch batch(batch_size, vector<float>(num_features, 1));
        float_batch target(batch_size, vector<float>(1, 1));
        for (int i = 0; i < batch_size; i++) {
            int idx = rand_int(gen);
            batch[i][0] = x[idx][0];
            batch[i][1] = x[idx][1];
            target[i][0] = t[idx][0];
        }

        y= my_net.forward(batch, false);
        Loss loss = celoss.apply(y, target);
        my_net.backward(loss);
        opt.apply();

        float reg_loss = 0;
        for(size_t i = 0; i < my_net.parameters.size(); i++){
            float norm2_W = 0;
            int w = my_net.parameters[i]->W.size(), h = my_net.parameters[i]->W[0].size();
            for (int k = 0; k < w; k++){
                for (int l = 0; l < h; l++){
                    norm2_W += pow(my_net.parameters[i]->W[k][l], 2);
                }
            }
            reg_loss += 0.5 * my_net.parameters[i]->lambda * norm2_W;
        }

        total_loss = loss.value + reg_loss;
        if (!smoothed_flag) {
            smoothed_loss = total_loss;
            smoothed_flag = true;
        } else {
            smoothed_loss = (float) (0.9 * smoothed_loss + 0.1 * total_loss);
        }
        if (step % 100 ==0)
            cout<<"Step: " << step <<" | loss: " << smoothed_loss<<endl;

    }
    y = my_net.forward(x, true);
    int predicted_class[num_samples * num_classes];
    for (int i = 0; i < num_samples * num_classes; i++) {
        int selected_class = -1;
        float max_prob = -std::numeric_limits<float>::max();
        for (int j = 0; j < num_classes; j++) {
            if (y[i][j] > max_prob) {
                max_prob = y[i][j];
                selected_class = j;
            }
        }
        predicted_class[i] = selected_class;
    }
    int true_positives = 0;
    for (int i = 0; i < num_samples * num_classes; i++) {
        if (predicted_class[i] == (int)t[i][0]) {
            true_positives++;
        }
    }
    cout<<"training acc: " << fixed << setprecision(0) << float(true_positives) / float(num_samples * num_classes) * 100 <<"%"<< endl;
}

void train_regression(){
    cout <<"-----Regression-------"<<endl;

    float_batch x(200, vector<float>(1, 1));
    float_batch t(200, vector<float>(1, 1));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 0.1};


    for(int i = -100; i < 100; i++){
        x[i + 100][0] = 0.01 * i;
        t[i + 100][0] = pow(x[i + 100][0], 2) + d(gen);

    }
    MyNet my_net = MyNet{1, 1};
    MSELoss mse{};
    //    Momentum opt(0.2, 0.8, my_net.parameters);
    //    SGD opt(0.3, my_net.parameters);
    //    RMSProp opt(0.01, 0.99, my_net.parameters);
    //    AdaGrad opt(0.05, my_net.parameters);
    Adam opt(0.003, 0.9, 0.999, my_net.parameters);
    float_batch y;

    for(int epoch = 0; epoch < 1000; epoch++){
        y= my_net.forward(x, false);
        Loss loss = mse.apply(y, t);
        my_net.backward(loss);
        opt.apply();
        if (epoch % 100 == 0)
            cout<<"Step: " << epoch <<" | loss: " << loss.value<<endl;
    }


    //    for(size_t i = 0; i < x.size(); i++){
    //        cout<<"x = " << x[i][0]<<endl;
    //        cout<<"t = " << t[i][0]<<endl;
    //        cout<<"y = " << y[i][0]<<endl;
    //        cout<< "delta = " << loss.delta[i][0]<<endl;
    //    }

    //    Utils utils;
    //    float_batch a = {{1, 2}, {3, 4}};
    //    float_batch b = {{5, 6}, {7, 8}};
    //    float_batch c = utils.mat_mul(a, b);

    //    for(int i = 0; i < 2; i++){
    //        for(int j = 0; j < 2; j++){
    //            cout<< a[i][j]<<endl;

    //            }
    //        }
}
