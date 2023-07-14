package Optimizers;
import Layers.*;

import java.util.ArrayList;

public abstract class Optimizer {
    float lr;
    public ArrayList<Layer> parameters;

    public Optimizer(float lr, ArrayList<Layer> params){
        this.lr = lr;
        this.parameters = params;
    }

    public abstract void apply();
}
