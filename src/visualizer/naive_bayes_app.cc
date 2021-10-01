#include <visualizer/naive_bayes_app.h>
#include <core/trainingdata.h>
#include <core/probabilitymodel.h>
#include <core/classification.h>

namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin) {
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

size_t NaiveBayesApp::ClassifyDrawnImage() const {
    // save the 2d vector as an image to classify
    // 1. load in training data from file
    // 2. save the data to a model
    naivebayes::TrainingData data = naivebayes::TrainingData(kImageSize);
    data.ReadInData("/Users/sana.madhavan/Desktop/Cinder/my-projects/naive-bayes-sm202073/data/trainingimagesandlabels.txt");

    // and saves the trained model to a file.
    naivebayes::ProbabilityModel model = naivebayes::ProbabilityModel(data, 1);

    // classify test images
    naivebayes::Classification classify = naivebayes::Classification(kImageSize);

    Image drawn_image = Image(sketchpad_.GetSketchpadPixels());
    drawn_image.SetSize(kImageSize);
    // make an image object
    classify.ReadInTestImage(drawn_image);

    return classify.FindClassificationClass(model);
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN:
      // ask your classifier to classify the image that's currently drawn on the
      // sketchpad and update current_prediction_

      current_prediction_ = ClassifyDrawnImage();

      break;

    case ci::app::KeyEvent::KEY_DELETE:
      sketchpad_.Clear();
      break;
  }
}

}  // namespace visualizer

}  // namespace naivebayes
