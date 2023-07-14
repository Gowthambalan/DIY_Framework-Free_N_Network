package Optimizers;

import Layers.Layer;
import Utils.*;
import java.util.ArrayList;

public class Adam extends Optimizer{
    float beta1, beta2;
    float eps = (float) Math.pow(10, -8);
    int k = 1;
    ArrayList<float[][]> mW = new ArrayList<>(),
            vW = new ArrayList<>(),
            mb = new ArrayList<>(),
            vb = new ArrayList<>();

    public Adam(ArrayList<Layer> params, float lr, float beta1, float beta2) {
        super(lr, params);
        this.beta1 = beta1;
        this.beta2 = beta2;
        for (Layer param : this.parameters) {
            this.mW.add(Utils.rescale(param.W, 0.0F));
            this.vW.add(Utils.rescale(param.W, 0.0F));
            this.mb.add(Utils.rescale(param.b, 0.0F));
            this.vb.add(Utils.rescale(param.b, 0.0F));
        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            this.mW.set(i, Utils.mat_add(Utils.rescale(this.parameters.get(i).dW,
                    1 - this.beta1), Utils.rescale(this.mW.get(i), this.beta1)));
            this.vW.set(i, Utils.mat_add(Utils.rescale(Utils.element_wise_mul(this.parameters.get(i).dW, this.parameters.get(i).dW),
                    1 - this.beta2), Utils.rescale(this.vW.get(i), this.beta2)));
            float[][] mW_hat = Utils.rescale(this.mW.get(i), 1 / (float)(1 - Math.pow(this.beta1, this.k)));
            float[][] vW_hat = Utils.rescale(this.vW.get(i), 1 / (float)(1 - Math.pow(this.beta2, this.k)));
            float[][] grad_step_w = Utils.element_wise_mul(mW_hat,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(vW_hat), this.eps)));
            this.parameters.get(i).W = Utils.mat_add(this.parameters.get(i).W, Utils.rescale(grad_step_w, -this.lr));

            this.mb.set(i, Utils.mat_add(Utils.rescale(this.parameters.get(i).db,
                    1 - this.beta1), Utils.rescale(this.mb.get(i), this.beta1)));
            this.vb.set(i, Utils.mat_add(Utils.rescale(Utils.element_wise_mul(this.parameters.get(i).db, this.parameters.get(i).db),
                    1 - this.beta2), Utils.rescale(this.vb.get(i), this.beta2)));
            float[][] mb_hat = Utils.rescale(this.mb.get(i), 1 / (float)(1 - Math.pow(this.beta1, this.k)));
            float[][] vb_hat = Utils.rescale(this.vb.get(i), 1 / (float)(1 - Math.pow(this.beta2, this.k)));
            float[][] grad_step_b = Utils.element_wise_mul(mb_hat,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(vb_hat), this.eps)));
            this.parameters.get(i).b = Utils.mat_add(this.parameters.get(i).b, Utils.rescale(grad_step_b, -this.lr));
        }
        this.k++;
    }
}
