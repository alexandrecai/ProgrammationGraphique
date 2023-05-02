package com.example.tp_opengl.blocks;

import android.opengl.Matrix;

import com.example.tp_opengl.Square;

public class OBlock {

    float[] firstSquarePos;
    Square[] squares;
    float[] squareColors;

    public OBlock(float[] firstSquarePos, float[] squareColors) {
        this.squares = new Square[4];
        Square square;
        float[] pos = firstSquarePos;
        this.squares[0] = new Square(pos,squareColors);
        pos[0] = firstSquarePos[0]+2.0f;
        this.squares[1] = new Square(pos,squareColors);
        pos[1] = firstSquarePos[1]+2.0f;
        this.squares[2] = new Square(pos,squareColors);
        pos[0] = firstSquarePos[0]-2.0f;
        this.squares[3] = new Square(pos,squareColors);
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

    }


}
