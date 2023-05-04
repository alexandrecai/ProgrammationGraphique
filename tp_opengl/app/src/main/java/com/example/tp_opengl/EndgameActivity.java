package com.example.tp_opengl;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

public class EndgameActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_endgame);

        Intent intentValues = getIntent();

        Button playAgain = findViewById(R.id.buttonPlayAgain);
        TextView scoreText = findViewById(R.id.textScore);
        int score = intentValues.getIntExtra("score", 0);
        scoreText.setText("SCORE : " + score);

        playAgain.setOnClickListener(view -> {
            Intent intent = new Intent(EndgameActivity.this, MenuActivity.class);
            startActivity(intent);
        });

    }
}