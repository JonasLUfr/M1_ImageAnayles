#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Fonction pour calculer l'histogramme
std::vector<int> calculateHistogram(const cv::Mat &image) {
    CV_Assert(image.channels() == 1); // Assurez-vous que l'image est en niveaux de gris
    std::vector<int> histogram(256, 0); // Initialisez 256 bins pour l'histogramme
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            uchar pixel = image.at<uchar>(y, x);
            histogram[pixel]++;
        }
    }
    return histogram;
}

// Fonction pour l'égalisation d'histogramme
cv::Mat histogramEqualization(const cv::Mat &image, std::vector<uchar> &lut) {
    std::vector<int> histogram = calculateHistogram(image);
    std::vector<int> cumulativeHistogram(256, 0);

    // Calculer l'histogramme cumulé
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }

    int totalPixels = image.rows * image.cols;

    // Générer la LUT (table de correspondance)
    lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uchar>((cumulativeHistogram[i] * 255.0) / totalPixels + 0.5);
    }

    // Appliquer la LUT
    cv::Mat equalizedImage = image.clone();
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            equalizedImage.at<uchar>(y, x) = lut[image.at<uchar>(y, x)];
        }
    }

    return equalizedImage;
}

// Fonction pour l'étirement d'histogramme
cv::Mat histogramStretching(const cv::Mat &image, std::vector<uchar> &lut) {
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);

    // Gérer le cas où minVal == maxVal
    if (maxVal == minVal) {
        cv::Mat stretchedImage = cv::Mat::ones(image.size(), CV_8UC1) * 128; // Image uniforme en gris
        lut.resize(256);
        for (int i = 0; i < 256; ++i) {
            lut[i] = static_cast<uchar>(i); // LUT d'identité
        }
        return stretchedImage;
    }

    // Créer la LUT pour l'étirement
    lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        if (i < minVal) {
            lut[i] = 0;
        } else if (i > maxVal) {
            lut[i] = 255;
        } else {
            lut[i] = static_cast<uchar>(((i - minVal) * 255.0) / (maxVal - minVal) + 0.5);
        }
    }

    // Appliquer la LUT
    cv::Mat stretchedImage = image.clone();
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            stretchedImage.at<uchar>(y, x) = lut[image.at<uchar>(y, x)];
        }
    }

    return stretchedImage;
}

// Fonction pour tracer et enregistrer l'histogramme
void plotAndSaveHistogram(const std::vector<int> &histogram, const std::string &title, const std::string &filename) {
    int histSize = 256;
    int histHeight = 400;
    int histWidth = 512;
    int binWidth = cvRound((double)histWidth / histSize);

    // Créer une image vide pour dessiner l'histogramme
    cv::Mat histImage(histHeight, histWidth, CV_8UC1, cv::Scalar(0));

    int maxHistValue = *std::max_element(histogram.begin(), histogram.end());
    std::vector<int> normalizedHist(histogram.size());
    for (size_t i = 0; i < histogram.size(); ++i) {
        normalizedHist[i] = static_cast<int>((histogram[i] * histHeight) / maxHistValue);
    }

    for (int i = 0; i < histSize; ++i) {
        cv::line(histImage,
                 cv::Point(binWidth * i, histHeight),
                 cv::Point(binWidth * i, histHeight - normalizedHist[i]),
                 cv::Scalar(255), 2);
    }

    // Afficher l'histogramme
    cv::imshow(title, histImage);

    // Enregistrer l'histogramme en tant qu'image
    cv::imwrite(filename, histImage);
}

// Fonction principale
int main() {
    // Charger une image en niveaux de gris
    cv::Mat image = cv::imread("Donnee1/lena.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
        return -1;
    }

    // Égalisation d'histogramme
    std::vector<uchar> equalizationLUT;
    cv::Mat equalizedImage = histogramEqualization(image, equalizationLUT);
    std::vector<int> equalizedHistogram = calculateHistogram(equalizedImage);

    // Étirement d'histogramme
    std::vector<uchar> stretchingLUT;
    cv::Mat stretchedImage = histogramStretching(image, stretchingLUT);
    std::vector<int> stretchedHistogram = calculateHistogram(stretchedImage);

    // Afficher et enregistrer les images
    cv::imshow("Image Originale", image);
    cv::imshow("Image Égalisée", equalizedImage);
    cv::imshow("Image Étirée", stretchedImage);
    cv::imwrite("equalized_image.png", equalizedImage);
    cv::imwrite("stretched_image.png", stretchedImage);

    // Afficher et enregistrer les histogrammes
    plotAndSaveHistogram(calculateHistogram(image), "Histogramme Original", "original_histogram.png");
    plotAndSaveHistogram(equalizedHistogram, "Histogramme Égalisé", "equalized_histogram.png");
    plotAndSaveHistogram(stretchedHistogram, "Histogramme Étiré", "stretched_histogram.png");

    // Attendre une touche
    cv::waitKey(0);

    return 0;
}
