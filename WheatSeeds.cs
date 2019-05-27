using System;
using System.Net.Http;
using System.Collections.Generic;

using NeuralNetLib;

namespace NeuralNetwork___Backpropagation
{
    public class WheatSeeds
    {
        public static void main()
        {
            HttpClient client = new HttpClient();

            string csvData = client.GetStringAsync("https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv").Result;
            string[] csvLines = csvData.Split(new char[] { '\n' });

            List<float[]> inputData = new List<float[]>();
            List<float[]> expectedResults = new List<float[]>();

            foreach (string line in csvLines)
            {
                string[] csvElements = line.Split(new char[] { ',' });
                List<float> dataPoints = new List<float>();

                string expectedClass = csvElements[csvElements.Length - 1];

                for (int i = 0; i < csvElements.Length - 1; i++)
                {
                    if (!String.IsNullOrEmpty(csvElements[i]))
                    {
                        dataPoints.Add(float.Parse(csvElements[i]));
                    }
                }

                inputData.Add(dataPoints.ToArray());

                switch (expectedClass)
                {
                    case "1": expectedResults.Add(new float[] { 1.0f, 0.0f, 0.0f }); break;
                    case "2": expectedResults.Add(new float[] { 0.0f, 1.0f, 0.0f }); break;
                    case "3": expectedResults.Add(new float[] { 0.0f, 0.0f, 1.0f }); break;
                }
            }

            // Normalize dataset

            int dataColumnCount = inputData[0].Length;

            float[,] columnMinMax = new float[dataColumnCount, 2];

            for (int i = 0; i < dataColumnCount; i++)
            {
                columnMinMax[i, 0] = float.MaxValue;
                columnMinMax[i, 1] = float.MinValue;
            }

            foreach (float[] data in inputData)
            {
                for (int column = 0; column < dataColumnCount; column++)
                {
                    columnMinMax[column, 0] = Math.Min(columnMinMax[column, 0], data[column]);
                    columnMinMax[column, 1] = Math.Max(columnMinMax[column, 1], data[column]);
                }
            }

            foreach (float[] data in inputData)
            {
                for (int column = 0; column < dataColumnCount; column++)
                {
                    data[column] = (data[column] - columnMinMax[column, 0]) / (columnMinMax[column, 1] - columnMinMax[column, 0]);
                }
            }

            // Classification
            // 7 5 3
            NeuralNetwork neuralNetwork = NeuralNetwork.Create(7, 3, 1, 5);

            // Learning
            Console.WriteLine("Learning: Dataset size: {0}", inputData.Count);

            for (int epoch = 0; epoch < 500; epoch++)
            {
                Console.WriteLine("Epoch {0}", epoch);

                Random random = new Random();
                List<int> usedIndices = new List<int>();

                //for (int i = 0; i < randomizedInputData.Count; i++)
                while (usedIndices.Count != inputData.Count)
                {
                    int index = (int)(random.NextDouble() * inputData.Count);

                    if (usedIndices.Contains(index))
                    {
                        continue;
                    }

                    usedIndices.Add(index);

                    List<float> result = neuralNetwork.FeedForward(inputData[index]);

                    neuralNetwork.BackPropagate(expectedResults[index]);

                    //Console.WriteLine("Processed: {0}", usedIndices.Count);
                }
            }

            // Validate
            float totalErrors = 0.0f;
            for (int i = 0; i < inputData.Count; i++)
            {
                List<float> result = neuralNetwork.FeedForward(inputData[i]);

                float error = neuralNetwork.GetErrorClassification(expectedResults[i]);

                totalErrors += error;

                Console.WriteLine("Result: {0}, {1}, {2} - Expected: {3}, {4}, {5} - Error: {6}", result[0], result[1], result[2], expectedResults[i][0], expectedResults[i][1], expectedResults[i][2], error);
            }

            Console.WriteLine("Total Errors: {0}", totalErrors);
            Console.WriteLine("Validation Dataset size: {0}", inputData.Count);
            Console.WriteLine("Total Accuracy: {0}", 100.0 - totalErrors / (float)inputData.Count * 100.0f);
        }
    }
}
