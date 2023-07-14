package Losses;

public abstract class LossFunc {
    float[][] pred, target;

    public LossFunc() {
    }

    public LossFunc(float[][] pred, float[][] target){
        this.pred = pred;
        this.target = target;
    }
    public abstract Loss apply(float[][] pred, float[][] target);
    public abstract float[][] delta();
}
