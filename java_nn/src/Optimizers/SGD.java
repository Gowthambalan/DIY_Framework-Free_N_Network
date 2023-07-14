package Optimizers;

import java.util.ArrayList;
import Layers.Layer;
import Utils.Utils;

public class SGD extends Optimizer {

    public SGD(float lr, ArrayList<Layer> params) {
        super(lr, params);
    }

    @Override
    public void apply() {
        for (Layer parameter : this.parameters) {
            parameter.W = Utils.mat_add(parameter.W, Utils.rescale(parameter.dW, -this.lr));
            parameter.b = Utils.mat_add(parameter.b, Utils.rescale(parameter.db, -this.lr));
        }
    }
}
