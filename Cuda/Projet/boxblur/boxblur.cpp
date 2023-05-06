#include <iostream>
#include <cmath>
#include <IL/il.h>

int main() {
    unsigned int image;
    ilInit();
    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage("in.jpg");

    int width, height, bpp, format;

    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    // Récupération des données de l'image
    unsigned char* data = ilGetData();

    // Création du tableau de sortie
    unsigned char* out = new unsigned char[width * height * bpp];

    // Paramètres du filtre
    int radius = 4;
    int kernelSize = 2 * radius + 1;
    float kernelSum = kernelSize * kernelSize;

    // Parcours de l'image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            // Initialisation des variables pour la somme des pixels dans le noyau
            float r_sum = 0.0f;
            float g_sum = 0.0f;
            float b_sum = 0.0f;

            // Parcours du noyau centré sur le pixel courant
            for (int j = -radius; j <= radius; ++j) {
                for (int i = -radius; i <= radius; ++i) {

                    // Calcul des coordonnées du pixel à ajouter à la somme
                    int xx = x + i;
                    int yy = y + j;

                    // Gestion des bords de l'image
                    xx = std::max(0, std::min(xx, width - 1));
                    yy = std::max(0, std::min(yy, height - 1));

                    // Récupération de la couleur du pixel à ajouter à la somme
                    int idx = (yy * width + xx) * bpp;
                    float r = data[idx];
                    float g = data[idx + 1];
                    float b = data[idx + 2];

                    // Ajout du pixel à la somme
                    r_sum += r;
                    g_sum += g;
                    b_sum += b;
                }
            }

            // Calcul de la couleur moyenne dans le noyau
            unsigned char r = static_cast<unsigned char>(r_sum / kernelSum);
            unsigned char g = static_cast<unsigned char>(g_sum / kernelSum);
            unsigned char b = static_cast<unsigned char>(b_sum / kernelSum);

            // Stockage de la couleur moyenne dans le tableau de sortie
            int idx = (y * width + x) * bpp;
            out[idx] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
        }
    }

    // Placement des données dans l'image
    ilTexImage(width, height, 1, bpp, format, IL_UNSIGNED_BYTE, out);

    // Sauvegarde de l'image
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);
    delete[] out;
    return 0;
}
