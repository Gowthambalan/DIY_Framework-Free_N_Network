package Optimizers;

import java.util.ArrayList;

import Layers.Layer;
import Utils.Utils;

public class Momentum extends Optimizer {
    float mu;
    ArrayList<float[][]> gW = new ArrayList<>(), gb = new ArrayList<>();

    public Momentum(ArrayList<Layer> params, float lr, float mu) {
        super(lr, params);
        this.mu = mu;
        for (Layer param : this.parameters) {
            this.gW.add(Utils.rescale(param.W, 0.0F));
            this.gb.add(Utils.rescale(param.b, 0.0F));

        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            this.gW.set(i, Utils.mat_add(this.parameters.get(i).dW, Utils.rescale(this.gW.get(i), this.mu)));
            this.parameters.get(i).W =
                    Utils.mat_add(this.parameters.get(i).W, Utils.rescale(this.gW.get(i), -this.lr));
            this.gb.set(i, Utils.mat_add(this.parameters.get(i).db, Utils.rescale(this.gb.get(i), this.mu)));
            this.parameters.get(i).b =
                    Utils.mat_add(this.parameters.get(i).b, Utils.rescale(this.gb.get(i), -this.lr));

        }
    }
}
