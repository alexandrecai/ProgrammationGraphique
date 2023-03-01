#include <iostream>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <math.h>

#include "../../Common/shaders_utilities.hpp"
#include "../../Common/lecture_trajectoire.hpp"
#include "lecture.hpp"

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;
using namespace std;

//Les tableaux pour les attributs des sommets
GLfloat *coordonnees;
GLfloat *couleurs;

// Les paramètres pour les interactions clavier et souris
float stepTrans = 1.0;
int mouseXOld, mouseYOld;
GLFWwindow* window;

double Xpos, Ypos;
bool leftbutton = false;
bool rightbutton = false;
bool middlebutton = false;

// Les différents identifiants VBO VAO et lancement des shaders
GLuint vboID[2];
GLuint vaoID;
GLuint programID;
GLuint MatrixID;

// Les différentes matrices
glm::mat4 Projection;
glm::mat4 View;
glm::mat4 Model;
glm::mat4 MVP;

glm::mat4 translation;
glm::mat4 trans_initial;
glm::mat4 rotation;


// Pour manipuler les différents fichiers
XDRFILE* xdrFile;
const char *trajectoire_file;
const char *couleur_file;


GLint NbAtoms; // le nombre d'atomes
GLfloat bbox[6]; // la bounding box
GLfloat centre[3]; // le centre de la bounding box
bool eof = false; // juste un booléen pour les boucles while...


void init() {


    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glEnable(GL_DEPTH_TEST);

    programID = LoadShaders("../../Shaders/TransformVertexShader.vert", "../../Shaders/ColorFragShader.frag");

    // Fonction (cf lecture.cpp) pour lire la première frame de la trajectoire
    // Elle permet d'initialiser
    // NbAtoms pour savoir le nombre d'atomes de chaque trajectoires
    // couleurs le tableau des attributs couleurs pour chaque sommet
    // coordonnees la tableau des coordonnées de chaque sommet
    // bbox la bounding box à  partir de xmin, xmax etc
    // centre le centre de la bounding box
    // Elle prend en paramètre deux fichiers pour une régle sur la couleur et le contenu de la trajectoire
    // trajectoire_file est un fichier .xtc avec un format compressé des données.
    // Dans le répertoire common il y a les fonctions nécessaires pour lire ce format de données.
    // get_first_frame fait juste un appel à ces fonctions....
    get_first_frame(&xdrFile, &NbAtoms, &couleurs, &coordonnees, bbox, centre, couleur_file, trajectoire_file);


    // La suite est maintenant habituelle
    // On gère tous les buffers nécessaires à transférer les attributs des sommets (coordonnées et couleurs).
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    glGenBuffers(2, vboID);

    glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
    glBufferData(GL_ARRAY_BUFFER, 3 * NbAtoms * sizeof(float), coordonnees, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboID[1]);
    glBufferData(GL_ARRAY_BUFFER, 3 * NbAtoms * sizeof(float), couleurs, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glEnableVertexAttribArray(1);


    // Mise en place de la matrice ModelViewProjection en lien avec "MVP" du vectex shader
    MatrixID = glGetUniformLocation(programID, "MVP");

    Projection = glm::perspective(70.0f, 4.0f / 3.0f, 0.1f, 1000.0f);
    View = glm::lookAt(glm::vec3(0, 0, 2 * bbox[5]), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));


    // On ajoute une translation initiale qui permet de positionner la molécule centrée en (0,0,0)
    // Pour ça on prend en compte le centre de la bounding box.
    trans_initial = glm::translate(glm::mat4(1.0f), glm::vec3(-centre[0], -centre[1], -centre[2]));

    Model = glm::mat4(1.0f);

    MVP = Projection * View * Model;

    rotation = glm::mat4(1.0f);

    translation = glm::mat4(1.0f);


}

void Display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(programID);

    Model = translation * rotation * trans_initial;

    MVP = Projection * View * Model;

    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);


    // On va utiliser une nouvelle primitive GL_POINTS avec une taille de 2
    // Attention en général on évite d'utiliser des points
    // Mais ça va être l'objectif du TD
    glPointSize(2);

    glDrawArrays(GL_POINTS, 0, NbAtoms);
    glfwSwapBuffers(window);

}

// Dans la boucle infini glut quand on a rien à faire on lit une
// nouvelle frame
void Idle() {

    get_frame(xdrFile,trajectoire_file,NbAtoms,coordonnees);
    glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
    glBufferData(GL_ARRAY_BUFFER, 3 * NbAtoms * sizeof(float), coordonnees, GL_DYNAMIC_DRAW);


}
static void Clavier(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key ==  GLFW_KEY_UP)
        translation = glm::translate(translation, glm::vec3(0.f, stepTrans, 0.f));
    if (key ==  GLFW_KEY_DOWN)
        translation = glm::translate(translation, glm::vec3(0.f, -stepTrans, 0.f));
    if (key ==  GLFW_KEY_RIGHT)
        translation = glm::translate(translation, glm::vec3(-stepTrans, 0.f, 0.f));
    if (key ==  GLFW_KEY_LEFT)
        translation = glm::translate(translation, glm::vec3(stepTrans, 0.f, 0.f));
    if (key ==  GLFW_KEY_PAGE_UP)
        translation = glm::translate(translation, glm::vec3(0.f, 0.f, stepTrans));
    if (key ==  GLFW_KEY_PAGE_DOWN)
     translation = glm::translate(translation, glm::vec3(0.f, 0.f, -stepTrans));
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void Souris(GLFWwindow *window, int button, int action, int mods) {

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            leftbutton=true;
            mouseXOld = Xpos;
            mouseYOld = Ypos;
        } else {
            leftbutton=false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            middlebutton=true;
            mouseXOld = Xpos;
            mouseYOld = Ypos;
        } else {
            middlebutton=false;
        }
    }

	if (button== GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mouseXOld = Xpos;
            mouseYOld = Ypos;
            rightbutton=true;
        }
        else {
            rightbutton=false;
        }
    }
}

void Motion(GLFWwindow *window, double xpos, double ypos)
{
    Xpos = xpos;
    Ypos = ypos;
    if (leftbutton==true) {
        rotation = glm::rotate(rotation, (float)(xpos-mouseXOld)*stepTrans/10, glm::vec3(0.f, 1.f, 0.f));
    }
    if (middlebutton==true) {
                translation = glm::translate(translation, glm::vec3(0.f, 0.f, -(ypos - mouseYOld) * stepTrans / 10));
                mouseXOld = xpos;
                mouseYOld = ypos;
    }
    if (rightbutton==true) {
        translation = glm::translate(translation, glm::vec3((xpos-mouseXOld)*stepTrans/10, -(ypos-mouseYOld)*stepTrans/10, 0.f));
        mouseXOld = xpos;
                mouseYOld = ypos;
    }
}


int main(int argc, char **argv) {

    trajectoire_file = argv[1];
    couleur_file = argv[2];


    glfwInit();

    // Quelle est la compatibilité OpenGL et contexte créé : ici OpenGL 4.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);


    // CORE_PROFILE on n'admet pas les fonctions OpenGL deprecated dans les versions précédentes  (même si elles sont encore disponibles)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // FORWARD_COMPATIBLE on n'admet pas les fonctions OpenGL qui vont devenir deprecated dans les futures versions ?
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Création de la fenetre
    window = glfwCreateWindow(500, 500, "Test Lumiere 1", NULL, NULL);

    glfwSetKeyCallback(window, Clavier);
    glfwSetMouseButtonCallback(window, Souris);
    glfwSetCursorPosCallback(window, Motion);

    if (!window) {
        std::cout << "Impossible de créer la fenêtre GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Taille et position à l'écran de la fenêtre créée
    glfwSetWindowPos(window, 100, 100);
    glfwSetWindowSize(window, 500, 500);

    // On attache le contexte actuel OpenGL à la fenêtre
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();

    init();
    while (!glfwWindowShouldClose(window)) {
        Display();
        glfwPollEvents();
        get_frame(xdrFile,trajectoire_file,NbAtoms,coordonnees);
        glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
        glBufferData(GL_ARRAY_BUFFER, 3 * NbAtoms * sizeof(float), coordonnees, GL_DYNAMIC_DRAW);
    }

    glfwTerminate();

    xdrfile_close(xdrFile);

    delete[] coordonnees;
    delete[] couleurs;
    return 0;

}
