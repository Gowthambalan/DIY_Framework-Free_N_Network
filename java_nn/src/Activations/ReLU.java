package Activations;

public class ReLU implements Activation{
    public float[][] forward(float[][] x) {
        int w = x.length, h = x[0].length;
        float[][] temp = new float[w][h];
        for(int i = 0; i < w; i++){
            for(int j = 0; j < h; j++){
                temp[i][j] = (x[i][j] > 0) ? x[i][j] : 0;
            }
        }
        return temp;
    }

    public float[][] derivative(float[][] x) {
        int w = x.length;
        int h = x[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = (x[i][j] > 0) ? 1 : 0;
            }
        }
        return temp;
    }
}