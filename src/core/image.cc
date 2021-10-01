//
// Created by Sana Madhavan on 4/3/21.
//

#include <core/image.h>

namespace naivebayes {

    Image::Image() {}

    Image::Image(const std::vector<std::vector<bool>> &img) {
        image_ = img;
    }

    Image::Image(const std::vector<std::vector<bool>> &img, size_t className, size_t size) {
        image_ = img;
        class_ = className;
        size_ = size;
    }

    std::istream &operator>>(std::istream &is, Image &image) {
        std::string line; // represents current line of input
        size_t line_count = 0;

        size_t row_index = 0; // which row of the image you're on
        // going thru an image
        while (std::getline(is, line) && line_count <= image.size_) {
            if (line.size() == 1 && line_count == 0) { // on the first line with
                image.class_ = std::stoi(line); // getting the class type of the image
                line_count++;
            } else if (line.size() == 0) {
                line_count++; // skips lines with only spaces
            } else {
                std::vector<bool> inner;
                inner.resize(line.size());
                image.image_.push_back(inner); // initializing the inner vector
                for (size_t i = 0; i < image.size_; i++) {
                    if (line[i] == ' ') {
                        image.image_[row_index][i] = false; // unshaded pixel
                        // row is line you're on
                        // column is how far across the line from 0 to 29 you are
                    } else if (line[i] == '#' || line[i] == '+') {
                        image.image_[row_index][i] = true; // shaded pixel
                    }
                }
                row_index++;
                line_count++;
            }
         }
        return is;
     }


    std::vector<std::vector<bool>> Image::GetImage() const {
        return image_;
    }

    size_t Image::GetClass() const {
        return class_;
    }

    void Image::SetClass(size_t className) {
        class_ = className;
    }

    void Image::SetSize(size_t size) {
        size_ = size;
    }

    size_t Image::GetSize() const {
        return size_;
    }

}