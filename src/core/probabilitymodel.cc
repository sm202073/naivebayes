//
// Created by Sana Madhavan on 4/3/21.
//

#include <core/probabilitymodel.h>

naivebayes::ProbabilityModel::ProbabilityModel() {}

naivebayes::ProbabilityModel::ProbabilityModel(const TrainingData& data, int laplaceConstant) {
    image_set_ = data;
    laplace_constant_ = laplaceConstant;
}

size_t naivebayes::ProbabilityModel::CountNumClassInImages(size_t classType) const {
    std::vector<Image> images = image_set_.GetImages(); // gets all the images
    // loop through number of images and count the number of images belonging to the specified class type
    size_t classCount = 0;
    for (const Image& img: images) {
        if (img.GetClass() == classType) {
            classCount++;
        }
    }
    return classCount;
}

double naivebayes::ProbabilityModel::CalculateClassProbability(size_t classType) const {
    size_t classCount = CountNumClassInImages(classType);
    // Applying P(class = c) formula
    double numerator = laplace_constant_ + classCount;
    double denominator = (10*laplace_constant_) + image_set_.GetImages().size();
    return numerator / denominator;
}

std::vector<std::vector<std::vector<double>>> naivebayes::ProbabilityModel::CalculatePixelProbabilities(size_t class_type, bool feature) const {

    std::vector<std::vector<double>> pixel_probabilities;
    std::vector<std::vector<std::vector<double>>> all_image_probabilities;

    std::vector<Image> images = image_set_.GetImages(); // gets all the images
    // loop through all the images
    size_t feature_count = 0;
    size_t class_count = 0;

    double numerator; // k + # of images belonging to class c where Fi,j = f
    double denominator = (2*laplace_constant_) + CountNumClassInImages(class_type);; // 2k + Total # of images belonging to class c

    for (const Image& img: images) {
        if (img.GetClass() == class_type) {
            class_count++;
            for (size_t i = 0; i < img.GetImage().size(); i++) {
                std::vector<double> pixelRows;
                pixelRows.resize(img.GetImage().at(i).size());
                pixel_probabilities.push_back(pixelRows);
                for (size_t j = 0; j < img.GetImage().size(); j++) {
                    if (img.GetImage()[i][j] == feature) {
                        feature_count++;
                        numerator = laplace_constant_ + feature_count;
                    }
                    pixel_probabilities[i][j] = (numerator / denominator);
                }
                feature_count = 0; // counts the images belonging to class c where Fi,j = shade (feature)
            }
            all_image_probabilities.push_back(pixel_probabilities);
        }
    }
    return all_image_probabilities;
}

double naivebayes::ProbabilityModel::GetProbabilityFromPixel(size_t class_type, bool feature, size_t row, size_t col) const {
    std::vector<std::vector<std::vector<double>>> all_probabilities = CalculatePixelProbabilities(class_type, feature);
    // loop through the vector of images with probabilities at their pixels
    double probability_to_return;
    for (const std::vector<std::vector<double>>& image: all_probabilities) {
        probability_to_return = image[row][col];
    }
    return probability_to_return;
}

std::ostream& operator<<(std::ostream& os, naivebayes::ProbabilityModel& model) {
    size_t class_number = 0;

    while (class_number <= 9) {
        for (const auto& probability: model.CalculatePixelProbabilities(class_number, true)) {
            for (size_t i = 0; i < probability.size(); i++) {
                for (size_t j = 0; i < probability.size(); j++) {
                    os << probability[i][j]; // prints out probability 2D array
                }
            }
        }

        for (const auto& probability: model.CalculatePixelProbabilities(class_number, false)) {
            for (size_t i = 0; i < probability.size(); i++) {
                for (size_t j = 0; i < probability.size(); j++) {
                    os << probability[i][j]; // prints out probability 2D array
                }
            }
        }
        class_number++; // prints out the next class images
    }
    return os;
}

naivebayes::ProbabilityModel naivebayes::ProbabilityModel::WriteOutProbabilities(const std::string& file_path) {
    std::ofstream output_file(file_path);
    if (output_file.is_open()) {
        output_file << *this;
        output_file.close();
    }
    return *this;
}

void naivebayes::ProbabilityModel::SetLaplaceConstant(size_t constant) {
    laplace_constant_ = constant;
}

size_t naivebayes::ProbabilityModel::GetLaplaceConstant() const {
    return laplace_constant_;
}

