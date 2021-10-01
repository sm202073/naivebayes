//
// Created by Sana Madhavan on 4/3/21.
//
#pragma once
#include <vector>
#include <sstream>
#include <fstream>
#include <string>

#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#endif //NAIVE_BAYES_IMAGE_H

namespace naivebayes {

    class Image {
    /**
    * Represents an image in the dataset and holds each pixel and whether it's shaded or unshaded.
    */
    public:
        Image();
        Image(const std::vector<std::vector<bool>>& img);
        Image(const std::vector<std::vector<bool>>& img, size_t className, size_t size);

        std::vector<std::vector<bool>> GetImage() const;

        /**
         * Operator overloading for the individual image
         * @param is istream object to carry out operator overloading
         * @param image to save the input into
         */
        friend std::istream& operator>>(std::istream& is, Image& image);

        void SetClass(size_t className);

        void SetSize(size_t size);

        size_t GetSize() const;

        size_t GetClass() const;

    private:
        // 2D vector representing individual image
        std::vector<std::vector<bool>> image_; // column and row indices holds the shade, booleans to represent if the image is shaded
        size_t class_; // class of the image
        size_t size_; // representing the size of the image
    };


} //namespace naivebayes