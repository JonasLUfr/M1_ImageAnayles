#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

/*
Utilisez la commande ici pour exécuter ce fichier:
g++ Image.cpp -o ImageAvecFiltrage `pkg-config --cflags --libs opencv4`
*/

//--------------------------
// Main Fonction de filtrage médian (implémentation manuelle)
// input: image en niveaux de gris
// kernelSize: taille du noyau du filtre (impair)
// Type uchar pour stocker les valeurs (unsigned char)
//--------------------------
cv::Mat medianFilter(const cv::Mat &input, int kernelSize) {
    CV_Assert(input.channels() == 1); // Simplification : ne traite que les images en niveaux de gris
    CV_Assert(kernelSize % 2 == 1);   // Accepte uniquement une taille de noyau impaire

    int Kradius = kernelSize / 2; // Rayon du noyau (moitié de la taille du noyau)
    cv::Mat output = input.clone(); // Clone pour initialiser la sortie avec les mêmes valeurs que l’image d’entrée

    // Évite de dépasser les bords de l’image lors de la récupération des pixels voisins
    for (int y = Kradius; y < input.rows - Kradius; y++) {
        for (int x = Kradius; x < input.cols - Kradius; x++) {
            // Utilisation d'un vecteur pour stocker les pixels du voisinage de (y, x)
            std::vector<uchar> neighbors;
            // Pré-allocation pour optimiser les performances
            neighbors.reserve(kernelSize * kernelSize);

            // Parcourir la région de voisinage autour du pixel (y, x)
            for (int j = -Kradius; j <= Kradius; j++) {
                for (int i = -Kradius; i <= Kradius; i++) {
                    neighbors.push_back(input.at<uchar>(y + j, x + i));
                }
            }
            // Trier les valeurs du voisinage (ordre croissant)
            std::sort(neighbors.begin(), neighbors.end()); 
            // La médiane est la valeur au centre du vecteur trié
            uchar medianVal = neighbors[neighbors.size()/2];
            // Affecter la valeur médiane au pixel (y, x) dans l’image de sortie
            output.at<uchar>(y, x) = medianVal;
        }
    }

    // Retourne l'image filtrée
    return output;
}

//--------------------------------
// Main Fonction de convolution (pour images en niveaux de gris)
// I * H, où H est un noyau de taille LxL
// input: image d'entrée (I), de type cv::Mat, en niveaux de gris
// kernel: noyau de convolution (H), également de type cv::Mat
// Retourne: l'image convoluée, de type cv::Mat
//--------------------------------
cv::Mat convolve(const cv::Mat &input, const cv::Mat &kernel) {
    CV_Assert(input.channels() == 1);
    // Vérifie que le noyau est une matrice carrée et que la taille est impaire
    CV_Assert(kernel.rows == kernel.cols && kernel.rows % 2 == 1);

    // Détermine le rayon du noyau pour définir la fenêtre glissante
    int Kradius = kernel.rows / 2;
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F); // Initialise une matrice de zéros pour stocker les résultats intermédiaires
    
    // Calcul de la convolution
    // Parcourir chaque pixel valide de l'image
    for (int y = Kradius; y < input.rows - Kradius; y++) {
        for (int x = Kradius; x < input.cols - Kradius; x++) {
            float sum = 0.0f;
            // Parcourir la région de voisinage centrée sur le pixel (y, x)
            for (int j = -Kradius; j <= Kradius; j++) {
                for (int i = -Kradius; i <= Kradius; i++) {
                    float val = static_cast<float>(input.at<uchar>(y + j, x + i));
                    float k = kernel.at<float>(j + Kradius, i + Kradius);
                    sum += val * k;
                }
            }
            output.at<float>(y, x) = sum;
        }
    }

    // Convertit les résultats en une image 8 bits
    cv::Mat finalOutput;
    output.convertTo(finalOutput, CV_8U, 1.0, 0.0);

    return finalOutput;
}

//------------------------
// Fonction de création d'un noyau moyen
// size: taille du noyau (impair)
// Objectif : calculer la moyenne des pixels voisins pour réduire le bruit
//------------------------
cv::Mat createMeanKernel(int size) {
    CV_Assert(size % 2 == 1);
    // Crée une matrice de taille size×size remplie de 1
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    // Divise chaque élément de la matrice par size×size pour normaliser
    kernel = kernel / (float)(size*size);
    return kernel;
}

//------------------------
// Fonction de création d'un noyau gaussien (version simplifiée)
// size: taille du noyau (impair)
// sigma: écart-type
//------------------------
cv::Mat createGaussianKernel(int size, float sigma) {
    CV_Assert(size % 2 == 1);
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    // Le rayon du noyau définit la taille de la région de voisinage
    int Kradius = size / 2;
    float sum = 0.0f;
    // Calcul des poids gaussiens
    for (int y = -Kradius; y <= Kradius; y++) {
        for (int x = -Kradius; x <= Kradius; x++) {
            // Formule discrète du noyau gaussien
            float val = std::exp(-(x*x + y*y)/(2*sigma*sigma));
            // Stocke les valeurs calculées dans la matrice correspondante
            kernel.at<float>(y+Kradius, x+Kradius) = val;
            sum += val;
        }
    }
    // Normalisation pour que la somme des éléments du noyau soit égale à 1
    kernel = kernel / sum;
    return kernel;
}

//------------------------
// Fonction de création d'un noyau Laplacien (3x3, version simplifiée)
// Utilisé principalement pour la détection des contours
//------------------------
cv::Mat createLaplacianKernel() {
    // Initialisation d'une matrice 3×3 avec les valeurs du noyau Laplacien
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        0,  1, 0,
        1, -4, 1,
        0,  1, 0);
    return kernel;
}

//------------------------
// Fonction de création d'un noyau Sobel (horizontal)
//------------------------
cv::Mat createSobelKernelX() {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);
    return kernel;
}

//------------------------
// Fonction de création d'un noyau Sobel (vertical)
//------------------------
cv::Mat createSobelKernelY() {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1);
    return kernel;
}

int main(int argc, char** argv) {
    // Interface de ligne de commande simple
    std::string imageName;
    std::cout << "Entrez le nom du fichier image dans le répertoire Donnee (par exemple, Autre.png) : ";
    std::cin >> imageName;

    // Ajouter le chemin du répertoire Donnee au nom du fichier
    std::string imagePath = "Donnee/" + imageName;

    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Erreur : Impossible de charger l'image : " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Veuillez sélectionner le type de filtre : \n";
    std::cout << "1: median\n2: mean\n3: gaussian\n4: laplacian\n5: sobelX\n6: sobelY\n";
    int choice;
    std::cin >> choice;

    cv::Mat result;
    std::string resultName;

    switch (choice) {
        case 1: {
            // Filtrage médian
            int ksize;
            std::cout << "Veuillez saisir la taille du noyau du filtre médian (impaire, par exemple 3) : ";
            std::cin >> ksize;
            result = medianFilter(input, ksize);
            resultName = "median_result.png";
            break;
        }
        case 2: {
            // Filtrage moyen
            int ksize;
            std::cout << "Veuillez saisir la taille moyenne du noyau du filtre (impaire, par exemple 3) : ";
            std::cin >> ksize;
            cv::Mat meanKernel = createMeanKernel(ksize);
            result = convolve(input, meanKernel);
            resultName = "mean_result.png";
            break;
        }
        case 3: {
            // Filtrage gaussien
            int ksize;
            float sigma;
            std::cout << "Veuillez saisir la taille du noyau du filtre gaussien (impaire, par exemple 5) : ";
            std::cin >> ksize;
            std::cout << "Veuillez saisir le sigma du noyau gaussien (par exemple 1,0) : ";
            std::cin >> sigma;
            cv::Mat gaussKernel = createGaussianKernel(ksize, sigma);
            result = convolve(input, gaussKernel);
            resultName = "gaussian_result.png";
            break;
        }
        case 4: {
            // Laplacien
            cv::Mat lapKernel = createLaplacianKernel();
            result = convolve(input, lapKernel);
            resultName = "laplacian_result.png";
            break;
        }
        case 5: {
            // Sobel X
            cv::Mat sobelXKernel = createSobelKernelX();
            result = convolve(input, sobelXKernel);
            resultName = "sobel_x_result.png";
            break;
        }
        case 6: {
            // Sobel Y
            cv::Mat sobelYKernel = createSobelKernelY();
            result = convolve(input, sobelYKernel);
            resultName = "sobel_y_result.png";
            break;
        }
        default:
            std::cerr << "Choix non valide." << std::endl;
            return -1;
    }

    // Afficher le résultat
    cv::imshow("Original", input);
    cv::imshow("Result", result);
    cv::waitKey(0);

    // Sauvegarde du résultat
    cv::imwrite(resultName, result);
    std::cout << "Les résultats de ce traitement ont été enregistrés en tant que : " << resultName << std::endl;

    return 0;
}
