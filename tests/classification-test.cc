//
// Created by Sana Madhavan on 4/12/21.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <core/probabilitymodel.h>
#include <core/classification.h>

using naivebayes::TrainingData;
using naivebayes::Classification;

TEST_CASE("Testing pixel probabilities with underflow", "[test][probability]") {

    SECTION("Pixel probabilities with underflowon 2 by 2") {
        // 1. Read in training data
        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(2);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculatePixelProbabilitiesWithUnderflow(0, model) == Approx(-1.961658506));
    }

    SECTION("Classifying 1 by 1 test image based on  by training and saving 2 by 2 training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(1);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/1by1testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculatePixelProbabilitiesWithUnderflow(1, model) == Approx(-0.40546510813));
    }

    SECTION("Classifying 5 by 5 test image based on  by training and saving 5 by 5 training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(5);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5trainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(5);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculatePixelProbabilitiesWithUnderflow(4, model) == Approx(1.2188212455));
    }

}

TEST_CASE("Testing likelihood scores of classifier on different image sizes", "[test][classify][likelihood]") {

    SECTION("Classifying 2 by 2 test image based on 2 by 2 training and saving 2 by 2 training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(2);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculateLikelihoodScore(model, 0) == Approx(-3.3479528671));
    }

    SECTION("Classifying 1 by 1 test image based on  by training and saving 2 by 2 training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(1);
        dataTest.ReadInData(
        "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(1);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/1by1testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculateLikelihoodScore(model, 1) == Approx(-2.1102132003));
    }

    SECTION("Classifying 5 by 5 test image based on  by training and saving 5 by 5 training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(5);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5trainingdata.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(5);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5testdata.txt");

        // 4. Gets the likelihood score
        REQUIRE(classifier.CalculateLikelihoodScore(model, 4) == Approx(-0.4859268468));
    }

}


TEST_CASE("Classifying different test images and getting accuracy", "[test][classify][accuracy]") {


    SECTION("Classifying test image of class 1 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass0.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 1);
    }

    SECTION("Classifying test image of class 0 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass1.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 0);
    }

    SECTION("Classifying test image of class 2 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3bt3testdataclass2.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 2);
    }

    SECTION("Classifying test image of class 3 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass3.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 3);
    }

    SECTION("Classifying test image of class 4 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass4.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 4);
    }

    SECTION("Classifying test image of class 5 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass5.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 5);
    }

    SECTION("Classifying test image of class 6 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass6.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 6);
    }

    SECTION("Classifying test image of class 7 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass7.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 7);
    }

    SECTION("Classifying test image of class 8 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass8.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 8);
    }

    SECTION("Classifying test image of class 9 based on training and saving subset training images") {

        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass9.txt");

        // 4. Gets the classification class from 0-9
        size_t class_type = classifier.FindClassificationClass(model);
        REQUIRE(class_type == 9);

    }


    SECTION("Accuracy of classification") {
        // 1. Read in training data
        TrainingData dataTest = TrainingData(3);
        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3trainingdataallclasses.txt");

        // 2. Saving the training data to a model
        naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(dataTest, 1);

        // 3. Creating a classifier and read in the test data with images of the same size
        Classification classifier = Classification(3);
        classifier.ReadInTestData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/3by3testdataclass9.txt");

        REQUIRE(classifier.GetClassifierAccuracy(model) > 0.7);
    }



}