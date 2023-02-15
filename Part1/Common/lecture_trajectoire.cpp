#include <iostream>
#include <fstream>

#include "lecture_trajectoire.hpp"

using namespace std;

int xdrfile_getframe_positions(int natoms, float *ap, XDRFILE* xfp)
{

    float precision;

    /*    char* ptr;
    int ndata;
    int i = xdrfile_read_char(ptr, ndata, xfp);
    cout << ptr << ndata << i <<endl;
    */

    if (xdrfile_read_compr_coord_float(ap, &natoms, &precision, xfp) != natoms) {
        std::cout << "Wrong atom count (not " << natoms << ')' << std::endl;
        return 1;
    }
    return 0;
}  

// xdrfile_getframe - get a complete frame
// returns 1 at the end of file
int xdrfile_getframe_header(int *natoms, int *step, float *tps, float *box, XDRFILE*   xfp)
{
    int magic;

    // Reading magic number
    if (xdrfile_read_int(&magic, 1, xfp) != 1) {
      //        std::cout << "Problem : magic or end of file" << std::endl;
        return 1; // EOF !
    }

    // Reading the number of atoms
    if (xdrfile_read_int(natoms, 1, xfp) != 1) {
        std::cout << "Problem : atoms" << std::endl;
        return 1; // EOF !
    }
    // Reading the step number
    if (xdrfile_read_int(step, 1, xfp) != 1) {
        std::cout << "Problem : step" << std::endl;
        return 1; // EOF !
    }

    // Reading the time
    if (xdrfile_read_float(tps, 1, xfp) != 1) {
        std::cout << "Problem : time" << std::endl;
        return 1; // EOF !
    }


    // Reading the bounding box of the simulation
    if (xdrfile_read_float(box, 9, xfp) != 9) {
      std::cout << "Problem reading 9 floats from the file" << std::endl;
      return 1;
    }

    return 0;
}

void choix_couleur(const char s, float *rgb) {
  switch(s){
  case 'H': rgb[0]=1.000;rgb[1]=1.000;rgb[2]=1.000; break;
  case 'C': rgb[0]=0.250;rgb[1]=0.750;rgb[2]=0.750; break;
  case 'N': rgb[0]=0.000;rgb[1]=0.000;rgb[2]=1.000; break;
  case 'O': rgb[0]=1.000;rgb[1]=0.000;rgb[2]=0.000; break;
  case 'S': rgb[0]=1.000;rgb[1]=1.000;rgb[2]=0.000; break;
  case 'P': rgb[0]=0.500;rgb[1]=0.500;rgb[2]=0.200; break;
  case 'B': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'I': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'L': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'K': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'R': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'M': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  case 'F': rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  default:  rgb[0]=0.500;rgb[1]=0.300;rgb[2]=0.000; break;
  }
}

