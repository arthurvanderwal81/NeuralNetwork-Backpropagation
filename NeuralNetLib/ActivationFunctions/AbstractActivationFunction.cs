using System.Collections.Generic;

namespace NeuralNetLib.ActivationFunctions
{
    public abstract class AbstractActivationFunction
    {
        public abstract float Calculate(float input);
        public abstract float[] Calculate(float[] input);
        public abstract float Derivative(float input);
        public abstract float[] Derivative(float[] input);
    }
}
