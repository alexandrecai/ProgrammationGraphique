package com.example.tp_opengl;



import android.content.Intent;
import android.media.MediaPlayer;
import android.os.Bundle;

import android.app.Activity;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;

/* Ce tutorial est issu d'un tutorial http://developer.android.com/training/graphics/opengl/index.html :
openGLES.zip HelloOpenGLES20
 */


public class OpenGLES30Activity extends Activity {

    // le conteneur View pour faire du rendu OpenGL
    private GLSurfaceView mGLView;

    public static final String TAG = "OpenGLES30Activity";

    public MediaPlayer mediaPlayer;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        mediaPlayer = MediaPlayer.create(this, R.raw.tetris_theme);
        mediaPlayer.setLooping(true);
        mediaPlayer.start();

        /* Création de View et association à Activity
           MyGLSurfaceView : classe à implémenter et en particulier la partie renderer */

        // On récupère la taille de la grille saisit par l'utilisateur

        Intent intent = getIntent();
        int nbColumnGrid = intent.getIntExtra("nbColumnGrid", 7);
        int nbRowGrid = intent.getIntExtra("nbRowGrid", 15);

        Log.d(TAG, String.valueOf(nbColumnGrid));
        Log.d(TAG, String.valueOf(nbRowGrid));

        /* Pour le plein écran */
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);


        mGLView = new MyGLSurfaceView(this, nbRowGrid, nbColumnGrid);
        /* Définition de View pour cette activité */
        setContentView(mGLView);
    }

    @Override
    protected void onStop() {
        super.onStop();
        mediaPlayer.release();
        mediaPlayer = null;
    }
}
