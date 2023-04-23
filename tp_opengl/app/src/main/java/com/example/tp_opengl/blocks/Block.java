package com.example.tp_opengl.blocks;

import com.example.tp_opengl.Square;

public abstract class Block {
    
    Square[] squares = {};

    float[] squareColors = {};

    public void display(){};

    public void rotate(){};

    public void hide(){};
}
