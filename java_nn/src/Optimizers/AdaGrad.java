package Optimizers;

import Layers.Dense;
import Layers.Layer;
import Utils.Utils;

import java.util.ArrayList;

public class AdaGrad extends Optimizer{
    float eps = (float) Math.pow(10, -8);
    ArrayList<float[][]> sW = new ArrayList<>(), sb = new ArrayList<>();

    public AdaGrad(ArrayList<Layer> params, float lr) {
        super(lr, params);
        for (Layer param : this.parameters) {
            this.sW.add(Utils.rescale(param.W, 0.0F));
            this.sb.add(Utils.rescale(param.b, 0.0F));

        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            float[][] grad_square_w = Utils.element_wise_mul(this.parameters.get(i).dW, this.parameters.get(i).dW);
            this.sW.set(i, Utils.mat_add(this.sW.get(i), grad_square_w));
            float[][] grad_step_w = Utils.element_wise_mul(this.parameters.get(i).dW,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(this.sW.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).W = Utils.mat_add(this.parameters.get(i).W, Utils.rescale(grad_step_w, -this.lr));

            float[][] grad_square_b = Utils.element_wise_mul(this.parameters.get(i).db, this.parameters.get(i).db);
            this.sb.set(i, Utils.mat_add(this.sb.get(i), grad_square_b));
            float[][] grad_step_b = Utils.element_wise_mul(this.parameters.get(i).db,
                    Utils.element_wise_rev(Utils.add_scalar(Utils.mat_sqrt(this.sb.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).b = Utils.mat_add(this.parameters.get(i).b, Utils.rescale(grad_step_b, -this.lr));
        }
    }
}

