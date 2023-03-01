#include "lecture.hpp"
#include <fstream>
#include <iostream>
using namespace std;

void get_first_frame(XDRFILE** xdrFile, int* NbAtoms, float** couleurs, float** coordonnees, float* bbox, float* centre, const char* couleur_file, const char* trajectoire_file)
{
    (*xdrFile) = xdrfile_open(trajectoire_file, "r"); //Ouverture du fichier en lecture

    float box[9]; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    float tps; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    int stepstep; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    if (xdrfile_getframe_header(NbAtoms, &stepstep, &tps, box, (*xdrFile))!=1) {
            // La lecture de l'entête permet de récupérer des informations générales sur les trajectoires
            // NbAtoms : nombre d'atomes pour chaque frame
            // On peut donc allouer une taille aux tableaux des attributs
            *coordonnees = new float[3*(*NbAtoms)*12];
            *couleurs = new float[3*(*NbAtoms)*12];
            char c;


            // gestion des couleurs à partir d'un fichier qui décrit une frame
            // Pour chaque atome on vient lire la couleur sous forme RGB
            ifstream f;
            float rgb[3];
            f.open(couleur_file,ios::in);
            int index = 0;
            f>>c;

            while (!f.eof()){
                choix_couleur(c,rgb);
                (*couleurs)[index] = rgb[0];
                (*couleurs)[index+1] = rgb[1];
                (*couleurs)[index+2] = rgb[2];
                index+=3;
                f>>c;
            }
            f.close();

            // Lecture des positions des atomes
            // le tableau coordonnees contient x,y,z pour chaque atome
            xdrfile_getframe_positions(*NbAtoms, *coordonnees, (*xdrFile));

            // construction de la bounding box afin de placer la caméra etc
            bbox[0] = (*coordonnees)[0];
            bbox[1] = (*coordonnees)[1];
            bbox[2] = (*coordonnees)[2];
            bbox[3] = (*coordonnees)[0];
            bbox[4] = (*coordonnees)[1];
            bbox[5] = (*coordonnees)[2];
            for (int i=1; i<*NbAtoms; i++) {
                if ((*coordonnees)[3*i]<bbox[0])
                    bbox[0] = (*coordonnees)[3*i];
                if ((*coordonnees)[3*i+1]<bbox[1])
                    bbox[1] = (*coordonnees)[3*i+1];
                if ((*coordonnees)[3*i+2]<bbox[2])
                    bbox[2] = (*coordonnees)[3*i+2];
                if ((*coordonnees)[3*i]>bbox[3])
                    bbox[3] = (*coordonnees)[3*i];
                if ((*coordonnees)[3*i+1]>bbox[4])
                    bbox[4] = (*coordonnees)[3*i+1];
                if ((*coordonnees)[3*i+2]>bbox[5])
                    bbox[5] = (*coordonnees)[3*i+2];
            }
        }
        centre[0] = bbox[0]+(bbox[3]-bbox[0])/2;
        centre[1] = bbox[1]+(bbox[4]-bbox[1])/2;
        centre[2] = bbox[2]+(bbox[5]-bbox[2])/2;

}

// Fonction supplémentaire pour lire une frame à l'infini
// cf idle() dans la boucle infinie de Glut...
void get_frame(XDRFILE *xdrFile, const char* trajectoire_file, int NbAtoms, float* coordonnees)
{
    float box[9]; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    float tps; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    int stepstep; // en paramètre de getframe_hearder mais on ne l'utilise pas ensuite
    if (xdrfile_getframe_header(&NbAtoms, &stepstep, &tps, box, xdrFile)!=1)
        xdrfile_getframe_positions(NbAtoms, coordonnees, xdrFile);
    else {
        // A la fin de la lecture on ferme le fichier
        // Et on l'ouvre à nouveau...
        //xdrfile_close(xdrFile);
        //xdrFile = xdrfile_open(trajectoire_file,"r");
    }
}
