//
// Created by Sana Madhavan on 4/3/21.
//
#pragma once
#include <vector>
#include "trainingdata.h"
#include <ostream>
#include <fstream>

#ifndef NAIVE_BAYES_PROBABILITYMODEL_H
#define NAIVE_BAYES_PROBABILITYMODEL_H

#endif //NAIVE_BAYES_PROBABILITYMODEL_H

namespace naivebayes {

    class ProbabilityModel {
    public:

        ProbabilityModel();
        ProbabilityModel(const TrainingData& data, int laplaceConstant);


    /**
    * Counts the number of images that have a certain class type in the images dataset
    * @param classType (0-9)
    * @return number of images that are of a specified class
     */
        size_t CountNumClassInImages(size_t classType) const;


        /**
         * Calculates the probability that the class is shaded or unshaded
         * @param classType the class number from 0-9
         * @param shade represents the shade of the class (false for unshaded, true for shaded)
         * @return the probabilities of each pixel in a certain class being shaded (two dimensional vector)
         */
        std::vector<std::vector<std::vector<double>>> CalculatePixelProbabilities(size_t classType, bool feature) const;

        /**
         * Caculate the probability at a particular pixel in the training images of a particular class
         * @param class_type
         * @param feature
         * @param row
         * @param col
         * @return
         */
         double GetProbabilityFromPixel(size_t class_type, bool feature, size_t row, size_t col) const;

         /**
         * Calculates the probability that the class is in the data set
         * @param classType the class number from 0-9
         * @return the probabilities of each pixel in a certain class being shaded (two dimensional vector)
         */
        double CalculateClassProbability(size_t classType) const;



        /**
         * Writes out all probability data into a file
         * @return the model object with all probabilities written
         */
        ProbabilityModel WriteOutProbabilities(const std::string& filePath);


        void SetLaplaceConstant(size_t constant);
        size_t GetLaplaceConstant() const;

    private:
        int laplace_constant_;
        TrainingData image_set_; // training data object representing the image set to draw from when calculating probabilities
    };

} //namespace naivebayes