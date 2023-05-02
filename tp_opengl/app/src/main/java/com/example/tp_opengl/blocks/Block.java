package com.example.tp_opengl.blocks;

import com.example.tp_opengl.Square;

public abstract class Block {

    float[] firstSquarePos;

    public Block(float[] firstSquarePos) {
        this.firstSquarePos = firstSquarePos;
    }

    Square[] squares = {};

    float[] squareColors = {};

    public void display(){
        for (Square square : squares) {
            //square.draw();
        }
    };

    public void rotate(){};

    public void hide(){};
}
