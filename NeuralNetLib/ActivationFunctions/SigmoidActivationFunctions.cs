using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetLib.ActivationFunctions
{
    public class SigmoidActivationFunctions : AbstractActivationFunction
    {
        private float Sigmoid(float x)
        {
            return (float)(1.0f / (1.0f + Math.Exp(-x)));
        }

        public override IList<float> Calculate(IList<float> input)
        {
            return input.Select(x => Sigmoid(x)).ToList();
        }

        public override IList<float> Derivative(IList<float> input)
        {
            List<float> result = new List<float>();

            foreach (float x in input)
            {
                float sigmoid = Sigmoid(x);

                result.Add(sigmoid * (1.0f - sigmoid));
            }

            return result;
        }
    }
}
