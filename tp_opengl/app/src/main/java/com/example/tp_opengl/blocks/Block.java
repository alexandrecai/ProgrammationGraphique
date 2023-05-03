package com.example.tp_opengl.blocks;

import com.example.tp_opengl.Square;

public interface Block {

    void display(float[] mMVPMatrix);

    void rotate();

    Square[] getSquares();

    void setSquares(Square[] squares);
}
