package com.example.tp_opengl;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

public class EndgameActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_endgame);

        Button playAgain = findViewById(R.id.buttonPlayAgain);

        playAgain.setOnClickListener(view -> {
            Intent intent = new Intent(EndgameActivity.this, MenuActivity.class);
            startActivity(intent);
        });

    }
}