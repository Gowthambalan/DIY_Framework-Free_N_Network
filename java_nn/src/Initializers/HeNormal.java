package Initializers;

import java.util.Random;

public class HeNormal {
    Random rand = new Random();
    String non_linearity, mode;
    public HeNormal(String non_linearity, String mode){
        this.rand.setSeed(1);
        this.non_linearity = non_linearity;
        this.mode = mode;
        if (!this.mode.equals("fan_in") && !this.mode.equals("fan_out")){
            throw new RuntimeException("Not supported mode!");
        }
    }
    public float[][] initialize(int fan_in, int fan_out){
        int fan = this.mode.equals("fan_in") ? fan_in : fan_out;
        float gain;
        if(this.non_linearity.equals("relu")){
            gain = (float)Math.sqrt(2);
        }
        else{
            throw new RuntimeException();
        }
        float std = (float)(gain / Math.sqrt(fan));
        float[][] temp = new float[fan_in][fan_out];
        for(int i = 0; i < fan_in; i++){
            for(int j=0; j < fan_out; j++){
                temp[i][j] = (float)(this.rand.nextGaussian() * std); // ~N(0, std)
            }
        }
        return temp;
    }
}
