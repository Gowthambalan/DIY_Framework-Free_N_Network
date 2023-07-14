package Initializers;

public class Constant {
    private final float c;

    public Constant(Float c) {
        this.c = c;
    }

    public float[][] initialize(int w, int h) {
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for(int j = 0; j < h; j++){
                temp[i][j] = this.c;
            }
        }
        return temp;
    }
}