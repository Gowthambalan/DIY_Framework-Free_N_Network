package Activations;

public class Linear implements Activation{
    public float[][] forward(float[][] x) {
        return x;
    }

    public float[][] derivative(float[][] x) {
        int w = x.length;
        int h = x[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = 1;
            }
        }
        return temp;
    }
}