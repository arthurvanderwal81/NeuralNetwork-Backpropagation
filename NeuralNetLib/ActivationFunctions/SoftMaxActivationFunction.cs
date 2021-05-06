using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkSharp.ActivationFunctions
{
    public class SoftMaxActivationFunction : AbstractActivationFunction
    {
        // https://keisan.casio.com/exec/system/15168444286206
        public override float Calculate(float input)
        {
            throw new NotImplementedException();
        }

        public override float[] Calculate(float[] input)
        {
            float shift = input.Max();
            List<float> result = input.Select(x => (float)Math.Exp(x - shift)).ToList();

            float expSum = result.Aggregate((acc, cur) => acc + cur);
            float oneOverExpSum = 1.0f / expSum;

            result = result.Select(x => x * oneOverExpSum).ToList();

            float resultSum = 0.0f;

            for (int i = 0; i < result.Count; i++)
            {
                resultSum += result[i];
            }

            return result.ToArray();
        }

        public override float Derivative(float input)
        {
            throw new NotImplementedException();
        }

        public override float[] Derivative(float[] input)
        {
            throw new NotImplementedException();
        }
    }
}
