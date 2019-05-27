using System.Collections.Generic;

namespace NeuralNetLib.ActivationFunctions
{
    public abstract class AbstractActivationFunction
    {
        public abstract IList<float> Calculate(IList<float> input);
        public abstract IList<float> Derivative(IList<float> input);
    }
}
