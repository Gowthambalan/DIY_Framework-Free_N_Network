package Initializers;
import java.util.Random;

public class XavierUniform {
    Random rand = new Random();
    public XavierUniform(){
        this.rand.setSeed(1);
    }
    public float[][] initialize(int fan_in, int fan_out){
        float std = (float)Math.sqrt(2.0 / (fan_in + fan_out));
        float a = (float)(std * Math.sqrt(3));

        float[][] temp = new float[fan_in][fan_out];
        for(int i = 0; i < fan_in; i++){
            for(int j=0; j < fan_out; j++){
                temp[i][j] = this.rand.nextFloat() * 2 * a - a; // ~U[-a, a]
            }
        }
        return temp;
    }
}
