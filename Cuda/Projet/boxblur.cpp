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

    // Traitement de l'image
    unsigned char* out_blur = new unsigned char[width * height];

    // Matrice de convolution 1/9
    float kernel[3][3] = {
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f}
    };

    int i, j, k, l;
    float sum;

    // Parcours de l'image
    for (j = 1; j < height - 1; ++j) {
        for (i = 1; i < width - 1; ++i) {

            sum = 0.0f;

            // Parcours de la matrice de convolution
            for (k = -1; k <= 1; ++k) {
                for (l = -1; l <= 1; ++l) {
                    sum += kernel[k+1][l+1] * data[((j+k)*width + (i+l))*bpp];
                }
            }

            out_blur[j*width + i] = (unsigned char)sum;
        }
    }

    // Placement des données dans l'image
    ilTexImage(width, height, 1, bpp, format, IL_UNSIGNED_BYTE, out_blur);

    // Sauvegarde de l'image
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage("out.jpg");

    // Libération de la mémoire
    ilDeleteImages(1, &image);
    delete[] out_blur;

    return 0;
}
