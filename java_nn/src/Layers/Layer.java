package Layers;

public abstract class Layer {
    public float[][] W, dW;
    public float[][] b, db;
    public float lam;
    public abstract float[][] forward(float[][] x, boolean eval);
    public abstract float[][] backward(float[][] x);
}



