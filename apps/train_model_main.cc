#include <iostream>

#include <core/trainingdata.h>
#include <core/probabilitymodel.h>
#include <core/classification.h>

int main() {

  // read in data from the full training file
    naivebayes::TrainingData data = naivebayes::TrainingData(29);
    data.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/trainingimagesandlabels.txt");

    // and saves the trained model to a file.
    naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(data, 1);

    // classify test images
    naivebayes::Classification classify = naivebayes::Classification(29);
    classify.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/testimagesandlabels.txt");

    // classify and get accuracy score
    std::cout << classify.GetClassifierAccuracy(model);



  return 0;

}
