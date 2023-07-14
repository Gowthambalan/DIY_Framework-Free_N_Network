package Initializers;

import java.util.Random;

public class RandomUniform implements Initializer {
    Random rand = new Random();
    public RandomUniform(){
        this.rand.setSeed(1);
    }
    public float[][] initialize(int w, int h){
        float[][] temp = new float[w][h];
        for(int i = 0; i < w; i++){
            for(int j=0; j < h; j++){
                temp[i][j] = this.rand.nextFloat();
            }
        }
        return temp;
    }
}
