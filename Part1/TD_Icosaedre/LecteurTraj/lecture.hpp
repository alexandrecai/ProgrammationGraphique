#ifndef LECTURE_HPP
#define LECTURE_HPP

#include "../../Common/lecture_trajectoire.hpp"

void get_first_frame(XDRFILE** xdrFile, int* NbAtoms, float** couleurs, float** coordonnees, float* bbox, float* centre, const char* couleur_file, const char* trajectoire_file);
void get_frame(XDRFILE *xdrFile, const char* trajectoire_file, int NbAtoms, float* coordonnees);

#endif
