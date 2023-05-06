#include <iostream>
#include <cmath>
#include <IL/il.h>

int main() {

    unsigned int image;

    ilInit();

    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage("../images/in.jpg");

    int width, height, bpp, format;

    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    // Récupération des données de l'image
    unsigned char* data = ilGetData();

    // Traitement de l'image
    unsigned char* out_grey = new unsigned char[width*height];
    unsigned char* out_gaussian = new unsigned char[width*height];


    // Conversion de l'image en niveaux de gris
    for (std::size_t i = 0; i < width*height; ++i) {
        out_grey[i] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;
    }

    unsigned int i, j, c;

    int total, res;


    for(j = 3 ; j < height - 3 ; ++j) {

        for(i = 3 ; i < width - 3 ; ++i) {

            // Horizontal
            total =   0 * out_grey[((j - 3) * width + i - 3) ] +  0 * out_grey[((j - 3) * width + i - 2) ] +   0 * out_grey[((j - 3) * width + i - 1) ] +   5 * out_grey[((j - 3) * width + i) ] +   0 * out_grey[((j - 3) * width + i + 1) ]  +  0 * out_grey[((j - 3) * width + i + 2) ] + 0 * out_grey[((j - 3) * width + i + 3) ]
                    + 0 * out_grey[((j - 2) * width + i - 3) ] +  5 * out_grey[((j - 2) * width + i - 2) ] +  18 * out_grey[((j - 2) * width + i - 1) ] +  32 * out_grey[((j - 2) * width + i) ] +  18 * out_grey[((j - 2) * width + i + 1) ]  +  5 * out_grey[((j - 2) * width + i + 2) ] + 0 * out_grey[((j - 2) * width + i + 3) ]
                    + 0 * out_grey[((j - 1) * width + i - 3) ] + 18 * out_grey[((j - 1) * width + i - 2) ] +  64 * out_grey[((j - 1) * width + i - 1) ] + 100 * out_grey[((j - 1) * width + i) ] +  64 * out_grey[((j - 1) * width + i + 1) ]  + 18 * out_grey[((j - 1) * width + i + 2) ] + 0 * out_grey[((j - 1) * width + i + 3) ]
                    + 5 * out_grey[((j) * width + i - 3) ]     + 32 * out_grey[((j) * width + i - 2) ]     + 100 * out_grey[((j) * width + i - 1) ]     + 100 * out_grey[((j) * width + i) ]     + 100 * out_grey[((j) * width + i + 1) ]      + 32 * out_grey[((j) * width + i + 2) ]     + 5 * out_grey[((j) * width + i + 3) ]
                    + 0 * out_grey[((j + 1) * width + i - 3) ] + 18 * out_grey[((j + 1) * width + i - 2) ] +  64 * out_grey[((j + 1) * width + i - 1) ] + 100 * out_grey[((j + 1) * width + i) ] +  64 * out_grey[((j + 1) * width + i + 1) ]  + 18 * out_grey[((j + 1) * width + i + 2) ] + 0 * out_grey[((j + 1) * width + i + 3) ]
                    + 0 * out_grey[((j + 2) * width + i - 3) ] +  5 * out_grey[((j + 2) * width + i - 2) ] +  18 * out_grey[((j + 2) * width + i - 1) ] +  32 * out_grey[((j + 2) * width + i) ] +  18 * out_grey[((j + 2) * width + i + 1) ]  +  5 * out_grey[((j + 2) * width + i + 2) ] + 0 * out_grey[((j + 2) * width + i + 3) ]
                    + 0 * out_grey[((j + 3) * width + i - 3) ] +  0 * out_grey[((j + 3) * width + i - 2) ] +   0 * out_grey[((j + 3) * width + i - 1) ] +   5 * out_grey[((j + 3) * width + i) ] +   0 * out_grey[((j + 3) * width + i + 1) ]  +  0 * out_grey[((j + 3) * width + i + 2) ] + 0 * out_grey[((j + 3) * width + i + 3) ]

                    ;


            //total = total > 255 ? 255 : total;
            //total = total < 0 ? 0 : total;

            res = total/1068;

            out_gaussian[(height - j - 1) * width + i] = res;

        }

    }

    // Placement des données dans l'image
    ilTexImage(width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_gaussian);

    // Sauvegarde de l'image
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);

    delete[] out_grey;
    delete[] out_gaussian;
}
