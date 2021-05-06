using System;
using System.Collections.Generic;

namespace NeuralNetLib.ActivationFunctions
{
    public class ReLuActivationFunction : AbstractActivationFunction
    {
        public override float Calculate(float input)
        {
            return Math.Max(0, input);
        }

        public override float[] Calculate(float[] input)
        {
            List<float> result = new List<float>();

            foreach(float x in input)
            {
                result.Add(Calculate(x));
            }

            return result.ToArray();
        }

        public override float Derivative(float input)
        {
            return input < 0.0f ? 0.0f : 1.0f;
        }

        public override float[] Derivative(float[] input)
        {
            List<float> result = new List<float>();

            foreach (float x in input)
            {
                result.Add(Calculate(x));
            }

            return result.ToArray();
        }
    }
}
