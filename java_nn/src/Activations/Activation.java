package Activations;

interface Activation {
    abstract float[][] forward(float[][] x);
    abstract public float[][] derivative(float[][] x);
}


