package Initializers;
import java.util.Arrays;


public class TestInitializers {

    public static void main(String[] args) {
        Constant constant = new Constant(10F);
        System.out.println(Arrays.deepToString(constant.initialize(1, 4)));

        RandomUniform random_uniform = new RandomUniform();
        System.out.println(Arrays.deepToString(random_uniform.initialize(3, 2)));


    }
}