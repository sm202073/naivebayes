//
// Created by Sana Madhavan on 4/4/21.
//

#define CATCH_CONFIG_MAIN
#include <core/trainingdata.h>
#include <catch2/catch.hpp>



using naivebayes::TrainingData;


TEST_CASE("Operator Overloading on Different Sizes", "[input][output][ifsteam][ostream]") {

    SECTION("Testing >> overloading on a 1 by 1 image") {

        TrainingData dataTest = TrainingData(1);

        dataTest.ReadInData(
                "/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/smallesttrainingdata.txt");

        std::vector<std::vector<bool>> imageTest(1, std::vector<bool>(1, true));
        std::vector<std::vector<bool>> actualImage = dataTest.GetImages().at(0).GetImage();
        REQUIRE(dataTest.GetImages().at(0).GetImage() == imageTest); // testing the only image in the dataset to check the full vector
    }

    SECTION("Testing >> overloading on 2 by 2 images") {

        TrainingData dataTest = TrainingData(2);
        dataTest.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/2by2trainingdata.txt");
        REQUIRE(dataTest.GetImages().size() == 2);

        std::vector<std::vector<bool>> imageTest(2, std::vector<bool>(2, true));
        REQUIRE(dataTest.GetImages().at(0).GetImage() == imageTest); // testing the first image

        std::vector<std::vector<bool>> imageTest2(2, std::vector<bool>(2, true));
        REQUIRE(dataTest.GetImages().at(1).GetImage() == imageTest); // testing the second image


    }

 SECTION("Testing >> overloading on a 5 by 5 image") {

        TrainingData dataTest = TrainingData(5);
        dataTest.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/5by5trainingdata.txt");
        REQUIRE(dataTest.GetImages().size() == 1);

        std::vector<bool> inner_one(5, true);
        std::vector<bool> inner_two = {false, true, true, true, true};
        std::vector<bool> inner_three = {true, true, false, true, true};
        std::vector<bool> inner_four(5, true);
        std::vector<bool> inner_five(5, true);

        std::vector<std::vector<bool>> imageTest;
        imageTest.push_back(inner_one);
        imageTest.push_back(inner_two);
        imageTest.push_back(inner_three);
        imageTest.push_back(inner_four);
        imageTest.push_back(inner_five);

        REQUIRE(dataTest.GetImages().at(0).GetImage() == imageTest); // only one  5 by 5 image in the set, which is equal to the hard coded image
 }

}