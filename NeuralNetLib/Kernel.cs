namespace NeuralNetLib
{
    public class Kernel
    {
        public float Bias { get; set; }
        public float[,,] Weights { get; set; } //private set
        public Kernel(int[] size)
        {
            Weights = new float[size[0], size[1], size[2]];
        }

        public float DotProduct(float[,,] input, int leftX, int topY)
        {
            float result = 0.0f;

            for (int channel = 0; channel < Weights.GetLength(0); channel++)
            {
                for (int y = 0; y < Weights.GetLength(1); y++)
                {
                    int inputY = y + topY;

                    for (int x = 0; x < Weights.GetLength(2); x++)
                    {
                        int inputX = x + leftX;

                        result += input[channel, inputY, inputX] * Weights[channel, y, x];
                    }
                }
            }

            return result + Bias;
        }
    }
}
