#include <catch2/catch.hpp>
#include <core/probabilitymodel.h>


using naivebayes::TrainingData;


TEST_CASE("Class identification and counting", "[class][label][count][probability]") {


    SECTION("Testing counting number of classes of a certain type on only 1 by 1 image") {

        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CountNumClassInImages(1) == 1); // one class in the subset with class of 1
    }

    SECTION("Testing counting number of classes on 2 by 2 images") {

        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CountNumClassInImages(0) == 2); // one class in the subset with class of 0
    }

    SECTION("Testing counting number of classes on 5 by 5 images") {

        TrainingData dataTest = TrainingData(5);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CountNumClassInImages(4) == 1); // one class in the subset with class of 0
    }

}

TEST_CASE("Class prior probability", "[class][label][count][probability]") {


    SECTION("Testing prior on only 1 by 1 image") {

        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CalculateClassProbability(1) == 2.0/11.0); // one class in the subset with class of 1
    }

    SECTION("Testing prior on 2 by 2 images") {

        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CountNumClassInImages(0) == 1); // one class in the subset with class of 0
    }

    SECTION("Testing prior 5 by 5 images") {

        TrainingData dataTest = TrainingData(5);
        dataTest.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        REQUIRE(model.CalculateClassProbability(4) ==  2.0/11.0); // one class in the subset with class of 0
    }

}


TEST_CASE("Testing mathematical correctness of shaded pixel probabilities", "[pixel][shaded][probabilities][data]") {

    SECTION("Testing pixel probabilities of shaded in 1 by 1") {

        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");

        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        // testing all pixels accuracy with class 1

        // testing probability that it's unshaded
        REQUIRE(model.GetProbabilityFromPixel(1, false, 0, 0) == 2.0 / 3.0);

    }

    SECTION("Testing pixel probabilities of shaded in 2 by 2") {
        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        // testing all pixels accuracy with class 0


        // testing probability that it's shaded
        REQUIRE(model.GetProbabilityFromPixel(0, true, 0, 0) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, true, 0, 1) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, true, 1, 0) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, true, 1, 1) == 2.0 / 3.0);

    }

}


TEST_CASE("Testing mathematical correctness of unshaded pixel probabilities", "[pixel][unshaded][probabilities][data]") {


    SECTION("Testing pixel probabilities of unshaded in 1 by 1") {

        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");

        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        // testing probability that it's shaded
        REQUIRE(model.GetProbabilityFromPixel(1, true, 0, 0) == 2.0 / 3.0);

    }

    SECTION("Testing pixel probabilities of unshaded in 2 by 2") {
        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);
        // testing probability that it's unshaded
        REQUIRE(model.GetProbabilityFromPixel(0, false, 0, 0) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, false, 0, 1) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, false, 1, 0) == 2.0 / 3.0);
        REQUIRE(model.GetProbabilityFromPixel(0, false, 1, 1) == 2.0 / 3.0);
    }
}


TEST_CASE("Testing << operator overloading", "[model][operator][overloading][write]") {


    SECTION("Testing writing out file of pixel probabilities of unshaded") {

        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        model.CalculatePixelProbabilities(1, true);


        // 1 is the only class the file holds
        REQUIRE(model.WriteOutProbabilities("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/modeloutput.txt").CalculateClassProbability(1) == (2.0/11.0));
    }

}



