using System;
using System.Collections.Generic;

namespace NeuralNetLib.ActivationFunctions
{
    public class ReLuActivationFunction : AbstractActivationFunction
    {
        public override IList<float> Calculate(IList<float> input)
        {
            List<float> result = new List<float>();

            foreach(float x in input)
            {
                result.Add(Math.Max(0, x));
            }

            return result;
        }

        public override IList<float> Derivative(IList<float> input)
        {
            List<float> result = new List<float>();

            foreach (float x in input)
            {
                result.Add(x < 0.0f ? 0.0f : 1.0f);
            }

            return result;
        }
    }
}
