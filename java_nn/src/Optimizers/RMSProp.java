package Optimizers;

import Layers.Layer;
import Utils.Utils;

import java.util.ArrayList;

public class RMSProp extends Optimizer {
    float beta;
    float eps = (float) Math.pow(10, -8);
    ArrayList<float[][]> sW = new ArrayList<>(), sb = new ArrayList<>();

    public RMSProp(ArrayList<Layer> params, float lr, float beta) {
        super(lr, params);
        this.beta = beta;
        for (Layer param : this.parameters) {
            this.sW.add(Utils.rescale(param.W, 0.0F));
            this.sb.add(Utils.rescale(param.b, 0.0F));

        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            float[][] grad_square_w = Utils.element_wise_mul(this.parameters.get(i).dW, this.parameters.get(i).dW);
            grad_square_w = Utils.rescale(grad_square_w, 1 - this.beta);
            this.sW.set(i, Utils.mat_add(Utils.rescale(this.sW.get(i), beta), grad_square_w));
            float[][] grad_step_w = Utils.element_wise_mul(this.parameters.get(i).dW,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(this.sW.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).W = Utils.mat_add(this.parameters.get(i).W, Utils.rescale(grad_step_w, -this.lr));

            float[][] grad_square_b = Utils.element_wise_mul(this.parameters.get(i).db, this.parameters.get(i).db);
            grad_square_b = Utils.rescale(grad_square_b, 1 - this.beta);
            this.sb.set(i, Utils.mat_add(Utils.rescale(this.sb.get(i), beta), grad_square_b));
            float[][] grad_step_b = Utils.element_wise_mul(this.parameters.get(i).db,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(this.sb.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).b = Utils.mat_add(this.parameters.get(i).b, Utils.rescale(grad_step_b, -this.lr));
        }
    }
}
