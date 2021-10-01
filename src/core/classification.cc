//
// Created by Sana Madhavan on 4/12/21.
//

#include <core/classification.h>

namespace naivebayes {

    Classification::Classification() {}

    Classification::Classification(size_t size) {
        test_image_size_ = size;
    }

    std::istream &operator>>(std::istream &is, Classification &classification) {
        std::string line;

        while (!(is.eof())) {
            Image curr_image = Image();
            curr_image.SetSize(classification.test_image_size_);
            is >> curr_image;
            classification.test_images_.push_back(curr_image);
        }
        return is;
    }

    Classification Classification::ReadInTestData(const std::string &file_path) {
        std::ifstream input_file(file_path);
        if (input_file.is_open()) {
            input_file >> *this;
            input_file.close();
        }
        return *this;
    }

    void Classification::ReadInTestImage(const Image& test) {
        test_images_.push_back(test);
    }


    double Classification::CalculatePixelProbabilitiesWithUnderflow(size_t class_type, const ProbabilityModel& model) const {
        // 1. loop through all images in the test data and see if the class matches, check if the pixel is shaded or unshaded starting from the first
        // 2. call the probability of the pixel in the training data

        double total_pixel_probability = 0; // holds the total probability that we are getting in this method
        for (const Image &img: test_images_) {
            // checking if the particular image is of the class type
            if (img.GetClass() == class_type) {
                for (size_t i = 0; i < img.GetImage().size(); i++) {
                    for (size_t j = 0; j < img.GetImage().size(); j++) {
                        total_pixel_probability += log(
                                model.GetProbabilityFromPixel(class_type, img.GetImage()[i][j], i, j));
                    }
                }
            }
        }
        return total_pixel_probability;
    }

    double Classification::CalculateLikelihoodScore(const ProbabilityModel& model, size_t class_type) const {
        double prior_probability_with_underflow = log(model.CalculateClassProbability(class_type)); // getting the prior probability
        double likelihood_score = prior_probability_with_underflow + CalculatePixelProbabilitiesWithUnderflow(class_type, model);
        return likelihood_score; // returns the overall likelihood score
    }

    size_t Classification::FindClassificationClass(const ProbabilityModel& model) const {

        double max_score = std::numeric_limits<int>::min();
        double current_score;
        size_t index_of_max_score;
        for (size_t i = 0; i < 9; i++) {
            current_score = CalculateLikelihoodScore(model, i);
            if (current_score > max_score) {
                max_score = current_score;
                index_of_max_score = i; // holds the class number with the highest score
            }
        }
        return index_of_max_score;
    }

    double Classification::GetClassifierAccuracy(const ProbabilityModel& model) const {
        size_t accuracy_counter = 0;
        for (const Image& img: test_images_) {
            if (FindClassificationClass(model) == img.GetClass()) {
                accuracy_counter++;
            }
        }
        return (accuracy_counter / test_images_.size());
    }

}
