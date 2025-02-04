 #include <iostream>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <math.h>

#include "../Common/shaders_utilities.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;

static GLfloat coordonnees[] = {
        0, 0, 0,
        0.5, 0, 0,
        0.5, 0.5, 0,
        0, 0.5, 0,
        0, 0, 0.5,
        0.5, 0, 0.5,
        0.5, 0.5, 0.5,
        0, 0.5, 0.5,
};

static GLfloat couleurs[] = {
        1, 1, 1,
        0, 1, 1,
        0, 0, 1,
        0, 1, 0,
        1, 1, 1,
        0, 1, 1,
        0, 0, 1,
        0, 1, 0
};

GLuint indices[] = {
        0, 2, 1,
        0, 1, 4,
        1, 4, 5,
        2, 1, 5,
        2, 1, 5,
        0, 4, 2
        
};


GLuint vboid[3];
GLuint programID;
GLuint vaoID;

// Pour les interactions, gestion des events.
GLFWwindow *window;
float stepTrans = 0.1;
int mouseXOld, mouseYOld;
double Xpos, Ypos;
bool leftbutton = false;
bool rightbutton = false;
bool middlebutton = false;

// Un identifiant encore ....
GLuint MatrixID;

// Des matrices ...
// On va utiliser pour les construire et les manipuler glm (OpenGL Mathematics)
glm::mat4 Projection; // une matrice de projection
glm::mat4 View; // une matrice de vue ?
glm::mat4 Model; // une matrice Model ?
glm::mat4 MVP; // Et le produit des trois

glm::mat4 translation; // une matrice de translation qui va nous permettre de déplacer le cube
glm::mat4 trans_initial; // une matrice de translation pour centrer le cube
glm::mat4 rotation; // une matrice pour construire une rotation appliquée sur le cube



void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glEnable(GL_DEPTH_TEST);

    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    programID = LoadShaders("../Shaders/TransformVertexShader.vert", "../Shaders/ColorFragShader.frag");

    // Petite fonction pour changer le mode de représentation (fil de fer ou plein)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    glGenBuffers(3, vboid);
    glBindBuffer(GL_ARRAY_BUFFER, vboid[0]);
    glBufferData(GL_ARRAY_BUFFER, 3 * 8 * sizeof(float), coordonnees, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glEnableVertexAttribArray(0);


    glBindBuffer(GL_ARRAY_BUFFER, vboid[1]);
    glBufferData(GL_ARRAY_BUFFER, 3 * 8 * sizeof(float), couleurs, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboid[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);



    // Il faut transmettre les informations à la carte graphique et au vertex shader
    // D'où cet identifiant (regarder dans TransformVertexShader et la déclaration d'une variable "uniform" MVP)
    MatrixID = glGetUniformLocation(programID, "MVP");

    // Création d'une matrice de translation pour déplacer le cube de -0.25 en X, Y et Z
    trans_initial = glm::translate(glm::mat4(1.0f), glm::vec3(-0.25, -0.25, -0.25));


    // On initialise Model à l'identité
    Model = glm::mat4(1.0f);

    // On parlera d'une matrice Model/View/Projection (MVP)
    // Pour le moment elle est réduite à la matrice Model soit déplacer l'objet dans le repère universel
    MVP = Model;

    // Initialisation d'une rotation et d'une translation à l'identité.
    rotation = glm::mat4(1.0f);
    translation = glm::mat4(1.0f);
}

void Display(void) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(programID);

    // Mise à jour de la matrice Model
    Model = translation * rotation * trans_initial;
    MVP = Model;
    // Transmission de la matrice à la carte graphique
    // Il est temps d'aller regarder le shader TransformVertexShader.vert
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

    glDrawElements(GL_TRIANGLES, 3 * 6, GL_UNSIGNED_INT, NULL);

    // Echange des buffers écriture de l'image et lecture (si double buffering)
    glfwSwapBuffers(window);
}

static void Clavier(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_UP)
        translation = glm::translate(translation, glm::vec3(0.f, stepTrans, 0.f));
    if (key == GLFW_KEY_DOWN)
        translation = glm::translate(translation, glm::vec3(0.f, -stepTrans, 0.f));
    if (key == GLFW_KEY_RIGHT)
        translation = glm::translate(translation, glm::vec3(stepTrans, 0.f, 0.f));
    if (key == GLFW_KEY_LEFT)
        translation = glm::translate(translation, glm::vec3(-stepTrans, 0.f, 0.f));
    if (key == GLFW_KEY_PAGE_UP)
        translation = glm::translate(translation, glm::vec3(0.f, 0.f, stepTrans));
    if (key == GLFW_KEY_PAGE_DOWN)
        translation = glm::translate(translation, glm::vec3(0.f, 0.f, -stepTrans));
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    glfwSwapBuffers(window);
}

void Souris(GLFWwindow *window, int button, int action, int mods) {

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            leftbutton = true;
            mouseXOld = Xpos;
            mouseYOld = Ypos;
        } else {
            leftbutton = false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            middlebutton = true;
            mouseXOld = Xpos;
            mouseYOld = Ypos;
        } else {
            middlebutton = false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mouseXOld = Xpos;
            mouseYOld = Ypos;
            rightbutton = true;
        } else {
            rightbutton = false;
        }
    }
}

void Motion(GLFWwindow *window, double xpos, double ypos) {
    Xpos = xpos;
    Ypos = ypos;
    if (leftbutton == true) {
        rotation = glm::rotate(rotation, (float) (xpos - mouseXOld) * stepTrans / 10, glm::vec3(0.f, 1.f, 0.f));
    }
    if (middlebutton == true) {
        //translation = glm::translate(translation, glm::vec3(0.f, 0.f, -(ypos - mouseYOld) * stepTrans / 10));
        //mouseXOld = xpos;
        //mouseYOld = ypos;
        rotation = glm::rotate(rotation, (float) (ypos - mouseYOld) * stepTrans / 10, glm::vec3(1.f, 0.f, 0.f));
    }
    if (rightbutton == true) {
        translation = glm::translate(translation, glm::vec3((xpos - mouseXOld) * stepTrans / 10,
                                                            -(ypos - mouseYOld) * stepTrans / 10, 0.f));
        mouseXOld = xpos;
        mouseYOld = ypos;
    }
}

int main(int argc, char **argv) {
    // On initialise GLFW : retourne un code d'erreur
    glfwInit();

    // Quelle est la compatibilité OpenGL et contexte créé : ici OpenGL 4.1
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);


    // CORE_PROFILE on n'admet pas les fonctions OpenGL deprecated dans les versions précédentes  (même si elles sont encore disponibles)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // FORWARD_COMPATIBLE on n'admet pas les fonctions OpenGL qui vont devenir deprecated dans les futures versions ?
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Création de la fenetre
    window = glfwCreateWindow(500, 500, "Hello World", NULL, NULL);

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


    // Fonction qui permet d'initialiser "des choses" si nécessaire (souvent nécessaire).
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();

    init();

    // Lancement de la boucle GLFW
    while (!glfwWindowShouldClose(window)) {
        // Appel de la fonction de dessin
        Display();
        // On récupère les events
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;

}
