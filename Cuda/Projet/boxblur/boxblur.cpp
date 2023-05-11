#include <iostream>
#include <cmath>
#include <IL/il.h>

int main() {

    unsigned int image;

    ilInit();

    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage("../images/mountain.jpg");

    int width, height, bpp, format;

    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    // Récupération des données de l'image
    unsigned char* data = ilGetData();

    // Traitement de l'image
    unsigned char* out_grey = new unsigned char[width*height];
    unsigned char* out_boxblur = new unsigned char[width*height];


    // Conversion de l'image en niveaux de gris
    for (std::size_t i = 0; i < width*height; ++i) {
        // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
        out_grey[i] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;
    }

    unsigned int i, j, c;

    int total, res;


    for(j = 1 ; j < height - 1 ; ++j) {

        for(i = 1 ; i < width - 1 ; ++i) {

            // Horizontal
            total =       out_grey[((j - 1) * width + i - 1) ] + out_grey[((j - 1) * width + i) ]  +  out_grey[((j - 1) * width + i + 1) ]
                    +  out_grey[( j      * width + i - 1) ] + out_grey[( j * width + i) ] + out_grey[( j * width + i +1 ) ]
                    +     out_grey[((j + 1) * width + i - 1) ] +  out_grey[( (j + 1) * width + i) ]  + out_grey[((j + 1) * width + i + 1) ];


            res = total/9;

            //out_boxblur[ j * width + i ] = sqrt(res);

            out_boxblur[(height - j - 1) * width + i] = res;

        }

    }

    // Placement des données dans l'image
    ilTexImage(width, height, 2, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_boxblur);

    // Sauvegarde de l'image
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);

    delete[] out_grey;
    delete[] out_boxblur;
}
