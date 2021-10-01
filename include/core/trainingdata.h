#pragma once
#include <string>
#include <core/image.h>
#include <sstream>
#include <fstream>

namespace naivebayes {

class TrainingData {

 public:

  TrainingData() = default;
  TrainingData(size_t size);

  // operator overloading
  friend std::istream& operator>>(std::istream& is, TrainingData& data);

  /**
   * Reads in the file path and saves the full training dataset into a vector of Images
   * @param filePath to the data set
   * @return the Training Data object which holds
   */
  TrainingData ReadInData(const std::string& filePath);



    std::vector<Image> GetImages() const;

    void SetImages(const std::vector<Image>& imageSet);

    void SetSize(size_t size);

    size_t GetSize() const;

  private:
    std::vector<Image> images_;
    size_t size_;
};

}  // namespace naivebayes