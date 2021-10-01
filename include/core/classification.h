//
// Created by Sana Madhavan on 4/12/21.
//

#include "probabilitymodel.h"

#ifndef NAIVE_BAYES_CLASSIFICATION_H
#define NAIVE_BAYES_CLASSIFICATION_H

#endif //NAIVE_BAYES_CLASSIFICATION_H

namespace naivebayes {
    /**
     * Class for classifying images into a particular class (week 2)
     * Holds functionality for classifying the model based on probability functions.
     */
    class Classification {
    public:
        Classification();
        Classification(size_t size);

        /**
        * Calculate classification class with highest probability
         * @param model holds the model with the saved data
         * @return a class type which corresponds to the class with the greatest likelihood score from 0-9
        */
        size_t FindClassificationClass(const ProbabilityModel& model) const;

        /**
         * Gets likelihood score based on a model and from a particular class
         * @param model to use prior probability functions
         * @param class_type the class from 0-9 of the image
         */
        double CalculateLikelihoodScore(const ProbabilityModel& model, size_t class_type) const;


        /**
         * Helper method to calculate pixel probabilities without demoninator and with logs
         * @param class_type represents the class that the probabilities are for
         * @param model represents the model to use to get individual pixel probabilities from the training data
         * @return the addition of the pixel probabilities with logs
         */
        double CalculatePixelProbabilitiesWithUnderflow(size_t class_type, const ProbabilityModel& model) const;

        /**
         * Reads in the test file with operator overloading
         * @param is istream object to handle operator overloading
         * @param classification object representing where the test file should read into
         */
        friend std::istream& operator>>(std::istream& is, Classification& classification);

        /**
         * Reads in a test image to classify
         * @param test image object that holds the 2D vector representing each pixel
         */
        void ReadInTestImage(const Image& test);


        /**
         * Calls the operator overloading to read in the test file
         * @param file_path to read in data from
         * @return Classification object with the read-in data
         */
        Classification ReadInTestData(const std::string& file_path);

        /**
         * Gets the accuracy of the classifier
         * @param model with the saved and trained data
         * @return a double representing the accuracy (number of correct classifications over the total)
         */
        double GetClassifierAccuracy(const ProbabilityModel& model) const;



        std::vector<Image> GetTestImages() const;



    private:
        size_t test_image_size_; // represents the size of test images
        std::vector<Image> test_images_; // holds all the test images in a vector
    };
}
