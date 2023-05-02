package com.example.tp_opengl;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class MenuActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

        Button littleGrid = findViewById(R.id.buttonLeft);
        Button bigGrid = findViewById(R.id.buttonRight);


        littleGrid.setOnClickListener(v -> {
            Intent intent = new Intent(MenuActivity.this, OpenGLES30Activity.class);
            intent.putExtra("nbColumnGrid", 7);
            intent.putExtra("nbRowGrid", 15);
            startActivity(intent);
        });

        bigGrid.setOnClickListener(v -> {
            Intent intent = new Intent(MenuActivity.this, OpenGLES30Activity.class);
            intent.putExtra("nbColumnGrid", 8);
            intent.putExtra("nbRowGrid", 16);
            startActivity(intent);
        });
    }
}