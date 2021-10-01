#include <core/trainingdata.h>


namespace naivebayes {

TrainingData::TrainingData(size_t size) {
    size_ = size;
}

std::istream& operator>>(std::istream& is, TrainingData& data) {
    std::string line;


    while (!(is.eof())) {
                Image curr_image = Image();
                curr_image.SetSize(data.size_);
                is >> curr_image;
                data.images_.push_back(curr_image);
    }

    return is;
}




TrainingData TrainingData::ReadInData(const std::string& filePath) {
    std::ifstream input_file(filePath);
    if (input_file.is_open()) {
        input_file >> *this;
        input_file.close();
    }
    return *this;
}


std::vector<Image> TrainingData::GetImages() const {
    return images_;
}

void TrainingData::SetImages(const std::vector<Image>& imageSet) {
    images_ = imageSet;
}

size_t TrainingData::GetSize() const {
    return size_;
}

void TrainingData::SetSize(size_t size) {
    size_ = size;
}

}  // namespace naivebayes