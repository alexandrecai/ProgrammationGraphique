#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <chrono>

int main() {

    unsigned int image;

    ilInit();

    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage("../images/carrot.jpg");

    int width, height, bpp, format;

    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


    // Récupération des données de l'image
    unsigned char* data = ilGetData();

    // Traitement de l'image
    unsigned char* out_grey = new unsigned char[width*height];
    unsigned char* out_edge_detection = new unsigned char[width*height];


    // Conversion de l'image en niveaux de gris
    for (std::size_t i = 0; i < width*height; ++i) {
        // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
        out_grey[i] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;
    }

    unsigned int i, j, c;

    int res;


    for(j = 2 ; j < height - 2 ; ++j) {

        for(i = 2 ; i < width - 2 ; ++i) {

            // Horizontal
            res =       out_grey[((j - 2) * width + i - 2) ] * 0 + out_grey[((j - 2) * width + i -1) ] * 0 +  out_grey[((j - 2) * width + i) ]* -1 + out_grey[((j - 2) * width + i +1 ) ] * 0 + out_grey[((j - 2) * width + i + 2) ] *0
                    +  out_grey[((j - 1) * width + i - 2) ] * 0 + out_grey[((j - 1) * width + i -1) ] * -1 +  out_grey[((j - 1) * width + i) ]* -2 + out_grey[((j - 1) * width + i +1 ) ] * -1 + out_grey[((j - 1) * width + i + 2) ] *0
                    +     out_grey[((j) * width + i - 2) ] * -1 + out_grey[((j) * width + i -1) ] * -2 +  out_grey[((j) * width + i) ]* 16 + out_grey[((j) * width + i +1 ) ] * -2 + out_grey[((j) * width + i + 2) ] * -1
                    +    out_grey[((j + 1) * width + i - 2) ] * 0 + out_grey[((j +1) * width + i -1) ] * -1 +  out_grey[((j + 1) * width + i) ]* -2 + out_grey[((j + 1) * width + i +1 ) ] * -1 + out_grey[((j + 1) * width + i + 2) ] *0
                    +    out_grey[((j + 2) * width + i - 2) ] * 0 + out_grey[((j + 2) * width + i -1) ] * 0 +  out_grey[((j + 2) * width + i) ]* -1 + out_grey[((j + 2) * width + i +1 ) ] * 0 + out_grey[((j + 2) * width + i + 2) ] *0;

            res = res > 255 ? 255 : res;
            res = res < 0 ? 0 : res;

            out_edge_detection[(height - j - 1) * width + i] = res;

        }

    }

    // Placement des données dans l'image
    ilTexImage(width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_edge_detection);

    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: 0." << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;

    // Sauvegarde de l'image
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);

    delete[] out_grey;
    delete[] out_edge_detection;
}
