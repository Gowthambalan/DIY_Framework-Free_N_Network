package Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
    public static float[][] mat_mul(float[][] A, float[][] B) {
        int n = A.length, m = A[0].length;
        int k = B.length, l = B[0].length;
        if (k != m) {
            throw new RuntimeException("Invalid shape for matrices!");
        }
        float[][] temp = new float[n][l];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                temp[i][j] = 0;
                for (int r = 0; r < m; r++) {
                    temp[i][j] += A[i][r] * B[r][j];
                }
            }
        }
        return temp;
    }

    public static float[][] element_wise_mul(float[][] A, float[][] B) {
        int n = A.length, m = A[0].length;
        int k = B.length, l = B[0].length;
        if (n != k || m != l) {
            throw new RuntimeException("Invalid shape for matrices!");
        }
        float[][] temp = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                temp[i][j] = A[i][j] * B[i][j];
            }
        }
        return temp;
    }

    public static float[][] mat_add(float[][] A, float[][] B) {
        int n = A.length, m = A[0].length;
        int k = B.length, l = B[0].length;
        if (n != k || m != l) {
            throw new RuntimeException("Invalid shape for matrices!");
        }
        float[][] temp = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                temp[i][j] = A[i][j] + B[i][j];
            }
        }
        return temp;
    }

    public static float[][] rescale(float[][] A, float scale) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = A[i][j] * scale;
            }
        }
        return temp;
    }


    public static float[][] transpose(float[][] A) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[h][w];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[j][i] = A[i][j];
            }
        }
        return temp;
    }

    public static float[][] element_wise_rev(float[][] A) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = 1 / A[i][j];
            }
        }
        return temp;
    }

    public static float[][] add_scalar(float[][] A, float scalar) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = A[i][j] + scalar;
            }
        }
        return temp;
    }

    public static float[][] mat_sqrt(float[][] A) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[w][h];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                temp[i][j] = (float) Math.sqrt(A[i][j]);
            }
        }
        return temp;
    }

    public static float[][] batch_mean(float[][] A) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[1][h];
        for (float[] a : A) {
            for (int j = 0; j < h; j++) {
                temp[0][j] += (a[j] / w);
            }
        }
        return temp;
    }

    public static float[][] batch_sum(float[][] A) {
        int h = A[0].length;
        float[][] temp = new float[1][h];
        for (float[] a : A) {
            for (int j = 0; j < h; j++) {
                temp[0][j] += a[j];
            }
        }
        return temp;
    }

    public static float[][] batch_var(float[][] A, float[][] mu) {
        int w = A.length, h = A[0].length;
        float[][] temp = new float[1][h];
        for (float[] a : A) {
            for (int j = 0; j < h; j++) {
                temp[0][j] += (Math.pow(a[j] - mu[0][j], 2) / w);
            }
        }
        return temp;
    }

    public static ArrayList<float[][]> equal_batch_size(float[][] A, float[][] B) {
        int w0 = A.length, w1 = B.length;
        int w;
        float[][] X, temp;
        if (w0 < w1){
            w = w1;
            temp = new float[w1][A[0].length];
            X = A;
        }
        else {
            w = w0;
            temp = new float[w0][B[0].length];
            X = B;
        }
        for(int i = 0; i < w; i++){
            temp[i] = X[0];
        }
        if (w0 < w1) {
            return new ArrayList<>(List.of(temp, B));
        }
        else{
            return new ArrayList<>(List.of(A, temp));
        }
    }

    public static void main(String[] args) {
        float[][] a = new float[][]{{1, 2}, {3, 4}};
        float[][] b = new float[][]{{5, 6}, {7, 8}};
        System.out.println(Arrays.deepToString(Utils.mat_mul(a, b)));
        Utils.mat_add(a, b);
        System.out.println(Arrays.deepToString(a));

    }
}
