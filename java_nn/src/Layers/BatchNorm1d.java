package Layers;

import Initializers.*;
import Utils.*;

import java.util.ArrayList;

public class BatchNorm1d extends Layer{

    float[][] mu, std;
    float[][] mu_hat, std_hat;
    float[][] x_hat, gamma;
    int in_features;
    float beta = 0.1F, eps = 0.00001F;
    Constant zeros = new Constant(0.0F);
    Constant ones = new Constant(1.0F);

    public BatchNorm1d(int in_features) {
        this.in_features = in_features;
        this.W = this.ones.initialize(1, this.in_features);
        this.b = this.zeros.initialize(1, this.in_features);
        this.mu_hat = this.zeros.initialize(1, this.in_features);
        this.std_hat = this.ones.initialize(1, this.in_features);
    }

    @Override
    public float[][] forward(float[][] x, boolean eval) {
        if (!eval){
            this.mu = Utils.batch_mean(x);
            this.std = Utils.mat_sqrt(Utils.batch_var(x, this.mu));
            this.mu_hat = Utils.mat_add(Utils.rescale(this.mu_hat, 1 - this.beta), Utils.rescale(this.mu, this.beta));
            this.std_hat = Utils.mat_add(Utils.rescale(this.std_hat, 1 - this.beta), Utils.rescale(this.std, this.beta));
        }
        else {
            this.mu = this.mu_hat;
            this.std = this.std_hat;
        }
        float[][] mu = new float[x.length][this.mu[0].length];
        float[][] std = new float[x.length][this.std[0].length];
        for (int i = 0; i < x.length; i++){
            mu[i] = this.mu[0];
            std[i] = this.std[0];
        }
        float[][] num = Utils.mat_add(x, Utils.rescale(mu, -1));
        float[][] den = Utils.mat_sqrt(Utils.add_scalar(Utils.element_wise_mul(std, std), this.eps));
        float[][] x_hat = Utils.element_wise_mul(num, Utils.element_wise_rev(den));
        this.x_hat = x_hat;

        this.gamma = new float[x.length][this.W[0].length];
        float[][] beta = new float[x.length][this.b[0].length];
        for (int i = 0; i < x.length; i++){
            this.gamma[i] = this.W[0];
            beta[i] = this.b[0];
        }
        float[][] y = Utils.mat_add(Utils.element_wise_mul(this.gamma, x_hat), beta);
        return y;
    }

    @Override
    public float[][] backward(float[][] delta) {
        float[][] dz = delta;
        float[][] dx_hat = Utils.element_wise_mul(dz, this.gamma);
        int m = dz.length;
        this.dW = Utils.rescale(Utils.batch_sum(Utils.element_wise_mul(this.x_hat, dz)), 1F / m);
        this.db = Utils.rescale(Utils.batch_sum(dz), 1F / m);

        float[][] a1 = Utils.rescale(dx_hat, m);
        float[][] a2 = Utils.batch_sum(dx_hat);
        ArrayList<float[][]> temp = Utils.equal_batch_size(this.x_hat, Utils.batch_sum(Utils.element_wise_mul(dx_hat, this.x_hat)));
        float[][] a3 = Utils.element_wise_mul(temp.get(0), temp.get(1));
        temp = Utils.equal_batch_size(Utils.rescale(a2, -1), Utils.rescale(a3, -1));
        float[][] num = Utils.mat_add(a1, Utils.mat_add(temp.get(0), temp.get(1)));
        float[][] den = Utils.rescale(Utils.mat_sqrt(Utils.add_scalar(Utils.element_wise_mul(this.std, this.std), this.eps)), m);

        temp = Utils.equal_batch_size(num , Utils.element_wise_rev(den));
        delta = Utils.element_wise_mul(temp.get(0), temp.get(1));
        return delta;
    }
}
