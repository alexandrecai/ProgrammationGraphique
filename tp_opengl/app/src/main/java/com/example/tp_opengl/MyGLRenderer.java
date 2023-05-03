/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.tp_opengl;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;

import com.example.tp_opengl.blocks.Block;
import com.example.tp_opengl.blocks.IBlock;
import com.example.tp_opengl.blocks.JBlock;
import com.example.tp_opengl.blocks.LBlock;
import com.example.tp_opengl.blocks.OBlock;
import com.example.tp_opengl.blocks.SBlock;
import com.example.tp_opengl.blocks.TBlock;
import com.example.tp_opengl.blocks.ZBlock;
import com.example.tp_opengl.constants.Colors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* MyGLRenderer implémente l'interface générique GLSurfaceView.Renderer */

public class MyGLRenderer implements GLSurfaceView.Renderer {

    private static final String TAG = "MyGLRenderer";

    private int nbRowGrid;
    private int nbColumnGrid;
    private final float squareSize = 1.0f;
    private final float[] mMVPMatrix = new float[16];

    private Block currentBlock;
    int nbRotation = 0;
    List<Square> allSquares = new ArrayList<>();





    /*
    Taille possible de grid (les plus communes) :
    7x15
    8x16
    10x20

     */

    public MyGLRenderer(int nbRowGrid, int nbColumnGrid) {
        Log.d(TAG, "Oh le beau renderer");
        this.nbRowGrid = nbRowGrid;
        this.nbColumnGrid = nbColumnGrid;


        Log.d("MyGLRenderer", String.valueOf(nbRowGrid));
        Log.d("MyGLRenderer", String.valueOf(nbColumnGrid));
    }


    private Square[][] grid;
/*
    private Square mSquare;
    private Square mSquare2;

 */

    // Les matrices habituelles Model/View/Projection

    private final float[] mProjectionMatrix = new float[16];
    private final float[] mViewMatrix = new float[16];

    private float[] mSquarePosition = {0.0f, 0.0f};


    /* Première méthode équivalente à la fonction init en OpenGLSL */
    @Override
    public void onSurfaceCreated(GL10 unused, EGLConfig config) {


        // la couleur du fond d'écran
        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    }

    /* Deuxième méthode équivalente à la fonction Display */
    @Override
    public void onDrawFrame(GL10 unused) {

        grid = new Square[nbColumnGrid + 2][nbRowGrid + 1];

        // glClear rien de nouveau on vide le buffer de couleur et de profondeur */
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);

        /* on utilise une classe Matrix (similaire à glm) pour définir nos matrices P, V et M*/

        /*Si on souhaite positionner une caméra mais ici on n'en a pas besoin*/
        // Matrix.setLookAtM(mViewMatrix, 0, 0, 0, -3, 0f, 0f, 0f, 0f, 1.0f, 0.0f);
         /* Pour le moment on va utiliser une projection orthographique
           donc View = Identity
         */
        Matrix.setIdentityM(mViewMatrix, 0);

        // Calculate the projection and view transformation
        Matrix.multiplyMM(mMVPMatrix, 0, mProjectionMatrix, 0, mViewMatrix, 0);

        // ##### GRID DISPLAY #####

        float[] gridColor = { // BLUE
                0.0f, 0.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 1.0f, 1.0f
        };
        float[] gridBorderColor = { // WHITE
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f
        };

        float[] gridSquarePosTest = {
                -2.0f,
                0.0f // On inverse l'axe y
        };

        float[] gridSquareMatrixTest = new float[16];

        float[] gridScratchTest = new float[16];


        float[] gridSquarePos2;
        try {
            gridSquarePos2 = new float[]{
                    this.currentBlock.getSquares()[0].get_position()[0],
                    this.currentBlock.getSquares()[0].get_position()[1]
            };
            this.currentBlock = new ZBlock(gridSquarePos2, Colors.blue, squareSize);

        } catch (Exception e) {
            gridSquarePos2 = new float[]{
                    (2*squareSize * 4) - ((grid.length - 1) * 1.0f),
                    -1.0f * (2*squareSize - ((grid[0].length - 1) * 1.0f))
            };
            createBlock();
        }

        for(Square square: this.allSquares){

            float[] gridSquareMatrix = new float[16];

            float[] gridScratch = new float[16];

            Matrix.setIdentityM(gridSquareMatrix, 0);

            Matrix.translateM(gridSquareMatrix, 0, square.get_position()[0], square.get_position()[1], 0);

            Matrix.multiplyMM(gridScratch, 0, mMVPMatrix, 0, gridSquareMatrix, 0);

            square.draw(gridScratch);

        }



        for (int i =0; i<nbRotation%4;i++){
            this.currentBlock.rotate();
        }

        this.currentBlock.display(mMVPMatrix);


        Log.d(TAG, "center x axis : " + (grid.length - 1));
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                //Log.d(TAG, "Grid : [" + i + "][" + j + "]");
                float[] gridSquareMatrix = new float[16];
                float[] gridSquarePos = {
                        (2*squareSize * i) - ((grid.length - 1) * 1.0f),
                        -1.0f * (2*squareSize * j - ((grid[0].length - 1) * 1.0f)) // On inverse l'axe y
                };

                // On met la bordure de la grille en blanc
                if (i == 0 || i == grid.length - 1 || j == grid[0].length - 1) {
                    grid[i][j] = new Square(gridSquarePos, Colors.white, squareSize);
                    //Log.d(TAG, "Grid Border : [" + i + "][" + j + "]");
                    float[] gridScratch = new float[16];

                    Matrix.setIdentityM(gridSquareMatrix, 0);

                    Matrix.translateM(gridSquareMatrix, 0, gridSquarePos[0], gridSquarePos[1], 0);

                    Matrix.multiplyMM(gridScratch, 0, mMVPMatrix, 0, gridSquareMatrix, 0);

                    //allSquares.add(grid[i][j]);

                    grid[i][j].draw(gridScratch);
                }
                /*
                else {
                    grid[i][j] = new Square(gridSquarePos, Colors.cyan);
                }


                 */


            }
        }

    }

    /* équivalent au Reshape en OpenGLSL */
    @Override
    public void onSurfaceChanged(GL10 unused, int width, int height) {
        /* ici on aurait pu se passer de cette méthode et déclarer
        la projection qu'à la création de la surface !!
         */
        GLES30.glViewport(0, 0, width, height);
        Matrix.orthoM(mProjectionMatrix, 0, -(width / 100.0f), width / 100.0f, -(height / 100.0f), height / 100.0f, -1.0f, 1.0f);

    }

    /* La gestion des shaders ... */
    public static int loadShader(int type, String shaderCode) {

        // create a vertex shader type (GLES30.GL_VERTEX_SHADER)
        // or a fragment shader type (GLES30.GL_FRAGMENT_SHADER)
        int shader = GLES30.glCreateShader(type);

        // add the source code to the shader and compile it
        GLES30.glShaderSource(shader, shaderCode);
        GLES30.glCompileShader(shader);

        return shader;
    }


    /* Les méthodes nécessaires à la manipulation de la position finale du carré */
    public void setPosition(float x, float y) {
        /*mSquarePosition[0] += x;
        mSquarePosition[1] += y;
        mSquarePosition[0] = x;
        mSquarePosition[1] = y;
*/
    }

    public float[] getPosition() {
        //return mSquarePosition;
        return new float[]{};
    }

    public void descendreBlock() {
        Square[] square = new Square[4];
        Square currentSquare;
        float[] pos;
        for (int i = 0; i < 4; i++) {
            pos = new float[]{
                    this.currentBlock.getSquares()[i].get_position()[0],
                    this.currentBlock.getSquares()[i].get_position()[1] - 2 * squareSize
            };
            currentSquare = new Square(pos, Colors.green, squareSize);
            square[i] = currentSquare;
            //System.out.println(currentSquare.get_position()[1]);
        }
        this.currentBlock.setSquares(square);
    }

    public void deplacerBlockGauche() {
        Square[] square = new Square[4];
        Square currentSquare;
        float[] pos;
        for (int i = 0; i < 4; i++) {
            pos = new float[]{
                    this.currentBlock.getSquares()[i].get_position()[0]-2*squareSize,
                    this.currentBlock.getSquares()[i].get_position()[1]
            };
            currentSquare = new Square(pos, Colors.green, squareSize);
            square[i] = currentSquare;
            //System.out.println(currentSquare.get_position()[1]);
        }
        this.currentBlock.setSquares(square);
    }

    public void deplacerBlockDroite() {
        Square[] square = new Square[4];
        Square currentSquare;
        float[] pos;
        for (int i = 0; i < 4; i++) {
            pos = new float[]{
                    this.currentBlock.getSquares()[i].get_position()[0] + 2*squareSize,
                    this.currentBlock.getSquares()[i].get_position()[1]
            };
            currentSquare = new Square(pos, Colors.green, squareSize);
            square[i] = currentSquare;
            //System.out.println(currentSquare.get_position()[1]);
        }
        this.currentBlock.setSquares(square);
    }


    public boolean isAtTheBottom(){
        Square bottom = grid[0][grid[0].length-1];
        float lowestSquareY = this.currentBlock.getSquares()[0].get_position()[1];
        for (int i = 1; i < 4; i++) {
            if (lowestSquareY> this.currentBlock.getSquares()[i].get_position()[1]){
                lowestSquareY = this.currentBlock.getSquares()[i].get_position()[1];
            }
        }
        if(lowestSquareY<=bottom.get_position()[1]+2*squareSize){
            allSquares.addAll(Arrays.asList(this.currentBlock.getSquares()));
            return true;
        }
        return false;
    }

    public void rotate() {
        // On calcule la position du centre de la barre
        this.nbRotation ++;
    }

    public void createBlock(){
        float[] initPos = new float[]{
                (2*squareSize * 4) - ((grid.length - 1) * 1.0f),
                -1.0f * (2*squareSize - ((grid[0].length - 1) * 1.0f))
        };
        this.currentBlock = new ZBlock(initPos,Colors.red,squareSize);
        this.nbRotation = 0;
    }



    public Block getCurrentBlock() {
        return currentBlock;
    }
}
