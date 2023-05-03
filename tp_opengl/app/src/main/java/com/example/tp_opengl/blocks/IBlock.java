package com.example.tp_opengl.blocks;

import android.opengl.Matrix;

import com.example.tp_opengl.Square;
import com.example.tp_opengl.constants.Colors;

public class IBlock implements Block{

    float[] firstSquarePos;
    Square[] squares;
    float[] squareColors;

    public IBlock(float[] firstSquarePos, float[] squareColors,float size) {
        this.squares = new Square[4];
        Square square;
        float[] pos = firstSquarePos;
        for(int i = 0;i<4;i++){
            if(i == 0){
                square = new Square(firstSquarePos, squareColors,size);
            }
            else {
                pos[0] = pos[0]+2*size;
                square = new Square(pos,squareColors,size);
            }
            this.squares[i] = square;
        }
        this.firstSquarePos = firstSquarePos;
        this.squareColors = squareColors;
    }



    public void display(float[] mMVPMatrix){

        float[] gridSquarePosTest;
        for (Square square : this.squares) {

            gridSquarePosTest = square.get_position();

            float[] gridSquareMatrixTest = new float[16];

            float[] gridScratchTest = new float[16];

            Matrix.setIdentityM(gridSquareMatrixTest,0);

            Matrix.translateM(gridSquareMatrixTest, 0, gridSquarePosTest[0], gridSquarePosTest[1], 0);

            Matrix.multiplyMM(gridScratchTest, 0, mMVPMatrix, 0, gridSquareMatrixTest, 0);

            square.draw(gridScratchTest);
        }
    }

    public void rotate() {
        // On calcule la position du centre de la barre
        float centerX = this.squares[2].get_position()[0];
        float centerY = this.squares[2].get_position()[1];

        // Pour chaque carré de la barre
        for (int i = 0; i < 4; i++) {
            Square square = squares[i];
            float[] pos = square.get_position();

            // On calcule la nouvelle position du carré en effectuant une rotation de 90 degrés autour du centre
            float newX = centerX + (pos[1] - centerY);
            float newY = centerY - (pos[0] - centerX);

            // On met à jour la position du carré
            square.set_position(new float[] {newX, newY});
        }
    }

    public Square[] getSquares() {
        return squares;
    }

    public void setSquares(Square[] squares) {
        this.squares = squares;
    }
}
