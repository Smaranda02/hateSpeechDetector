
using Microsoft.ML;
using Microsoft.ML.Data;
//using SentimentAnalysis;
//using sentimentanalysis;
using static Microsoft.ML.DataOperationsCatalog;
using System.IO;
using System.Net.Http;
using System.Net;



class Program
{
    static async Task Main()
    {

        string url = "http://localhost:5003/dataset";
        string fileName = "dataset.txt";

        //string filePath = "D:\\ANUL 2\\SEM2\\METODE DEZ SOFT\\ML_PROJECT\\dataset_hate_speech.csv";
        //string filePath = "dataset_hate_speech.csv";

        string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "dataset_hate_speech.csv");



        //string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", fileName);

        string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "datasetWorking.txt");

        // string _dataPath2 = Path.Combine(Environment.CurrentDirectory, "Data", "sentiments.txt");



        using (HttpClient client = new HttpClient())
        {
            using (var formContent = new MultipartFormDataContent())
            {
                using (var fileStream = File.OpenRead(filePath))
                {
                    var fileContent = new StreamContent(fileStream);
                    formContent.Add(fileContent, "file", Path.GetFileName(filePath));

                    HttpResponseMessage response = await client.PostAsync(url, formContent);

                    if(response.IsSuccessStatusCode)
                    {
                        Console.WriteLine("File uploaded successfully.");

                        string responseText = await response.Content.ReadAsStringAsync();
                        //Console.WriteLine("Response text: " + responseText);

                        string responseFilePath = "D:\\source\\repos\\SentimentAnalysis\\SentimentAnalysis\\Data\\dataset.txt";
                        

                        try
                        {
                            File.WriteAllText(responseFilePath, responseText);
                            Console.WriteLine("Response saved to: " + responseFilePath);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine("Error writing response to file: " + ex.Message);
                        }
                    }


                    else
                    {
                        // Handle the request error
                        Console.WriteLine("Error occurred: " + response.StatusCode);
                    }


                }
            }
        }


        MLContext mlContext = new MLContext();

        TrainTestData splitDataView = LoadData(mlContext);

        ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

        Evaluate(mlContext, model, splitDataView.TestSet);

        UseModelWithSingleItem(mlContext, model);

        UseModelWithBatchItems(mlContext, model);

        TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }


        ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText)).Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "It s none of your business"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Hate speech / Offensive " : "Not offensive")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
        new SentimentData
        {
            SentimentText = "You bitch is ugly"
        },
        new SentimentData
        {
            SentimentText = "Piece of shit"
        }
    };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Hate speech / Offensive " : "Not offensive")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }


    }
}