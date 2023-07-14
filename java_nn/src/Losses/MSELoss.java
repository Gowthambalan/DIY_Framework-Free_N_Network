package Losses;
import java.lang.Math;
import Utils.Utils;

public class MSELoss extends LossFunc{
    Utils utils = new Utils();

    public MSELoss() {
    }

    public MSELoss(float[][] pred, float[][] target) {
        super(pred, target);
    }

    @Override
    public Loss apply(float[][] pred, float[][] target){
        this.pred = pred;
        this.target = target;
        int w = pred.length, h = pred[0].length;
        float loss = 0;
        for(int i = 0; i < w; i++){
            for(int j = 0; j < h; j++){
                loss += Math.pow(pred[i][j] - target[i][j], 2) / 2;
            }
        }
        return new Loss(loss / w, this.delta());
    }

    @Override
    public float[][] delta(){
        float[][] delta = this.utils.mat_add(this.pred, this.utils.rescale(this.target, -1));
        return delta;
    }

}
