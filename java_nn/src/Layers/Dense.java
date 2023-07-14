package Layers;

import Initializers.*;
import Utils.*;
import Activations.*;

public class Dense extends Layer {

    int in_features;
    int out_features;
    String weight_initializer = "xavier_uniform";
    String bias_initializer = "zeros";
    String regularizer_type = null;
    public float lam = 0.0F;
    Linear linear = new Linear();
    ReLU relu = new ReLU();
    String act_name = "linear";
    XavierUniform xavier_uniform = new XavierUniform();
    RandomUniform random_uniform = new RandomUniform();
    Constant zeros = new Constant(0.0F);
    float[][] input, z;

    public Dense(int in_features,
                 int out_features,
                 String activation,
                 String weight_initializer,
                 String bias_initializer,
                 String regularizer_type,
                 float lam) {
        this.in_features = in_features;
        this.out_features = out_features;
        this.act_name = activation;
        this.weight_initializer = weight_initializer;
        this.bias_initializer = bias_initializer;
        this.regularizer_type = regularizer_type;
        this.lam = lam;

        HeNormal he_normal = new HeNormal(activation, "fan_in");

        if (weight_initializer.equals("random_uniform")) {
            this.W = this.random_uniform.initialize(this.in_features, this.out_features);
        } else if (weight_initializer.equals("xavier_uniform")) {
            this.W = this.xavier_uniform.initialize(this.in_features, this.out_features);
        } else {
            this.W = he_normal.initialize(this.in_features, this.out_features);
        }
        if (bias_initializer.equals("zeros")) {
            this.b = this.zeros.initialize(1, this.out_features);
        }

    }

    @Override
    public float[][] forward(float[][] x, boolean eval) {
        this.input = x;
        float[][] z = Utils.mat_mul(x, this.W);
        float[][] b = new float[z.length][this.b[0].length];
        for (int i = 0; i < z.length; i++) {
            b[i] = this.b[0];
        }
        z = Utils.mat_add(z, b);
        this.z = z;

        float[][] a = (this.act_name.equals("relu")) ? this.relu.forward(z) : this.linear.forward(z);
        return a;
    }

    @Override
    public float[][] backward(float[][] delta) {
        float[][] dz;
        if (this.act_name.equals("relu")) {
            dz = Utils.element_wise_mul(delta, this.relu.derivative(this.z));
        } else {
            dz = Utils.element_wise_mul(delta, this.linear.derivative(this.z));
        }

        float[][] input_t = Utils.transpose(this.input);
        float[][] dw = Utils.mat_mul(input_t, dz);
        this.dW = Utils.rescale(dw, 1F / dz.length);

        if (this.regularizer_type.equals("l2")){
            this.dW = Utils.mat_add(this.dW, Utils.rescale(this.W, this.lam));
        }
        else if (this.regularizer_type.equals("l1")){
            this.dW = Utils.add_scalar(this.dW, this.lam);
        }

        float[][] ones_t = new float[1][dz.length];
        for (int i = 0; i < ones_t.length; i++) {
            for (int j = 0; j < ones_t[0].length; j++) {
                ones_t[i][j] = 1;
            }
        }

        float[][] db = Utils.mat_mul(ones_t, dz);
        this.db = Utils.rescale(db, 1F / dz.length);

//        if (this.regularizer_type.equals("l2")){
//            this.db = Utils.mat_add(this.db, Utils.rescale(this.b, this.lam));
//        }
//        else if (this.regularizer_type.equals("l1")){
//            this.db = Utils.add_scalar(this.db, this.lam);
//        }

        float[][] w_t = Utils.transpose(this.W);
        delta = Utils.mat_mul(dz, w_t);
        return delta;
    }

    public static void main(String[] args) {
        float[][] a = new float[][]{{1, 2}, {3, 4}};
        float[][] b = new float[][]{{1, 2}, {3, 4}};

    }
}
