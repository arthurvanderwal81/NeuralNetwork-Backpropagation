using System;

using NeuralNetLib.Helpers;

namespace NeuralNetLib
{
    public class Kernel
    {
        public float Bias { get; set; }
        public float[,,] Weights { get; private set; }
        public float[,,] ErrorSignals { get; set; }

        public Kernel(int[] size)
        {
            Weights = new float[size[0], size[1], size[2]];
            ErrorSignals = new float[size[0], size[1], size[2]];

            for (int channel = 0; channel < size[0]; channel++)
            {
                for (int y = 0; y < size[1]; y++)
                {
                    for (int x = 0; x < size[2]; x++)
                    {
                        Weights[channel, y, x] = RandomHelper.GetRandomNormalDistributionFloat();//.GetRandomFloat(-1.0f, 1.0f);
                    }
                }
            }
        }

        public float DotProduct(Array input, int leftX, int topY)
        {
            float[,,] inputValues = input as float[,,];
            float result = 0.0f;

            for (int channel = 0; channel < Weights.GetLength(0); channel++)
            {
                for (int y = 0; y < Weights.GetLength(1); y++)
                {
                    int inputY = y + topY;

                    for (int x = 0; x < Weights.GetLength(2); x++)
                    {
                        int inputX = x + leftX;

                        result += inputValues[channel, inputY, inputX] * Weights[channel, y, x];
                    }
                }
            }

            return result + Bias;
        }

        public void ResetErrorSignals()
        {
            for (int channel = 0; channel < Weights.GetLength(0); channel++)
            {
                for (int y = 0; y < Weights.GetLength(1); y++)
                {
                    for (int x = 0; x < Weights.GetLength(2); x++)
                    {
                        ErrorSignals[channel, y, x] = 0.0f;
                    }
                }
            }
        }

        public void AddErrorSignal(float error, float derivative)
        {
            for (int channel = 0; channel < Weights.GetLength(0); channel++)
            {
                for (int y = 0; y < Weights.GetLength(1); y++)
                {
                    for (int x = 0; x < Weights.GetLength(2); x++)
                    {
                        ErrorSignals[channel, y, x] += error * Weights[channel, y, x] * derivative;
                    }
                }
            }
        }

        public void UpdateWeights(float learningRate)
        {
            for (int channel = 0; channel < Weights.GetLength(0); channel++)
            {
                for (int y = 0; y < Weights.GetLength(1); y++)
                {
                    for (int x = 0; x < Weights.GetLength(0); x++)
                    {
                        Weights[channel, y, x] += learningRate * ErrorSignals[channel, y, x];

                        Bias += learningRate * ErrorSignals[channel, y, x];
                    }
                }
            }
        }
    }
}
