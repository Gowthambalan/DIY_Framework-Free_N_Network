package Losses;


public class CrossEntropyLoss extends LossFunc {

    public CrossEntropyLoss() {
    }

    @Override
    public Loss apply(float[][] pred, float[][] target) {
        this.pred = pred;
        this.target = target;
        int w = pred.length;
        float[][] probs = this.soft_max(pred);
        float loss = 0;
        for (int i = 0; i < w; i++) {
            loss += -Math.log(probs[i][(int)target[i][0]]);
        }
        return new Loss(loss / w, this.delta());
    }

    @Override
    public float[][] delta() {
        int w = this.pred.length, h = this.pred[0].length;
        float[][] probs = this.soft_max(this.pred);
        for (int i = 0; i < w; i++) {
            probs[i][(int)this.target[i][0]] -= 1;
        }
        return probs;
    }

    public float[][] soft_max(float[][] x) {
        int w = x.length, h = x[0].length;
        float[][] num = new float[w][h];
        float[] den = new float[w];
        for (int i = 0; i < w; i++) {
            float max_of_batch = -Float.MAX_VALUE;
            float sum_of_batch = 0;
            for (int j = 0; j < h; j++) {
                if (x[i][j] > max_of_batch) {
                    max_of_batch = x[i][j];
                }
            }
            for (int j = 0; j < h; j++) {
                num[i][j] = (float) Math.exp(x[i][j] - max_of_batch);
                sum_of_batch += num[i][j];
            }
            den[i] = sum_of_batch;
        }
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                num[i][j] = num[i][j] / den[i] + (float) Math.pow(10, -6);
            }
        }
        return num;
    }
}
