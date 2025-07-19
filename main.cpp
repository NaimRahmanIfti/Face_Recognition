#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

int main() {
    // 1. open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) 
    {
        std::cerr << "ERROR: could not open camera\n";
        return 1;
    }
    // 2. load face cascade
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("data/haarcascade_frontalface_default.xml"))
    {
        std::cerr << "ERROR: could not load cascade\n";
        return 1;
    }

    // 3. load LBPH model
    auto model = cv::face::LBPHFaceRecognizer::create(
        2, // radius
        16, // neighbors
        8, // grid_x
        8, // grid_y
        75.0 // threshold
    );
    model->read("data/face_model.yml");
    // 4. load the label map
    std::map<int, std::string> labels;
    {
        std::ifstream in("data/labels.txt");
        if (!in) 
        {
            std::cerr << "ERROR: could not open labels.txt\n";
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) 
        {
            std::istringstream ss(line);
            int id; char sep;
            std::string name;
            if (ss >> id >> sep && std::getline(ss, name)) 
            {
                labels[id] = name;
            }
        }
    }
    cv::Mat frame, gray;
    std::vector<cv::Rect> faces;

    // 5) real‐time loop
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        // detect
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, {30,30});

        // for each face: recognize
        for (auto& r : faces) 
        {
            // 1) enlarge the box slightly
            r.x      = std::max(0, r.x - 10);
            r.y      = std::max(0, r.y - 10);
            r.width  = std::min(frame.cols - r.x, r.width + 20);
            r.height = std::min(frame.rows - r.y, r.height + 20);
    
            // 2) crop & resize
            cv::Mat faceROI = gray(r);
            cv::resize(faceROI, faceROI, cv::Size(200,200));
    
            // 3) predict with LBPH
            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(faceROI, predictedLabel, confidence);
    
            // 4) apply your confidence threshold
            double thresh = 75.0;
            std::string name = "Unknown";
            if (labels.count(predictedLabel) && confidence < thresh) {
                name = labels[predictedLabel];
            }
    
            // 5) draw
            cv::rectangle(frame, r, cv::Scalar(0,255,0), 2);
            cv::putText(frame,
                        name + " (" + cv::format("%.1f", confidence) + ")",
                        {r.x, r.y - 5},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0,255,0),
                        2);
        }
        cv::imshow("Face Recognition", frame);
        if (cv::waitKey(30) == 27) break;  // ESC to quit
    }

    return 0;
}
