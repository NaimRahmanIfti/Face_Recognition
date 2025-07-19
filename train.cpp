#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <iostream>
#include <map>         // for std::map
#include <fstream>     // for std::ofstream


namespace fs = std::filesystem;

int main() {
    const std::string train_dir = "data/train";
    const std::string model_file = "data/face_model.yml";
    // Data holders
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    std::map<int, std::string> label_names;
    int nextLabel = 0;
    // Check if the training directory exists
    if (!fs::exists(train_dir)) {
        std::cerr << "Training directory does not exist: " << train_dir << "\n";
        return 1;
    }
    // Iterate over each person folder
    for (auto& personEntry : fs::directory_iterator(train_dir)) {
        if (!personEntry.is_directory()) continue;
        std::string personName = personEntry.path().filename().string();
        int personLabel = nextLabel++;
        label_names[personLabel] = personName;

        // Load each image file in that folder
        for (auto& imgEntry : fs::directory_iterator(personEntry.path())) {
            if (imgEntry.path().extension() != ".png" &&
                imgEntry.path().extension() != ".jpg" &&
                imgEntry.path().extension() != ".jpeg")
                continue;
            cv::Mat img = cv::imread(imgEntry.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Warning: could not read "
                          << imgEntry.path() << "\n";
                continue;
            }
            // (Optional) resize all faces to a common size
            cv::resize(img, img, cv::Size(200, 200));

            images.push_back(img);
            labels.push_back(personLabel);
        }
    }
    if (images.empty()) {
        std::cerr << "No training images found in " << train_dir << "\n";
        return 1;
    }

    // Create & train the LBPH model
    // Create & train the LBPH model with custom params
    auto model = cv::face::LBPHFaceRecognizer::create(
        2, // radius
        16, // neighbors
        8, // grid_x
        8, // grid_y
        75.0 // threshold
    );

    model->read("data/face_model.yml");

    // Save trained model and label map
    model->save(model_file);
    std::cout << "Model trained and saved to " << model_file << "\n";

    // (Optional) write label map
    std::ofstream mapOut("data/labels.txt");
    for (auto& kv : label_names) {
        mapOut << kv.first << ";" << kv.second << "\n";
    }
    std::cout << "Labels written to data/labels.txt\n";

    return 0;
}
