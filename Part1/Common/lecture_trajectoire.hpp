#ifndef LECTEURTRAJ_HPP
#define LECTEURTRAJ_HPP

#include "xdrfile.h"

int xdrfile_getframe_positions(int natoms, float *ap, XDRFILE* xfp);
int xdrfile_getframe_header(int *natoms, int *step, float *tps, float *box, XDRFILE*   xfp);
void choix_couleur(const char s, float *rgb);
#endif
