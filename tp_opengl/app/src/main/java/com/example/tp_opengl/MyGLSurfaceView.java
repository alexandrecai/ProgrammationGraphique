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

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.MotionEvent;

import java.util.Timer;
import java.util.TimerTask;

/* La classe MyGLSurfaceView avec en particulier la gestion des événements
  et la création de l'objet renderer

*/


/* On va dessiner un carré qui peut se déplacer grâce à une translation via l'écran tactile */

public class MyGLSurfaceView extends GLSurfaceView {

    /* Un attribut : le renderer (GLSurfaceView.Renderer est une interface générique disponible) */
    /* MyGLRenderer va implémenter les méthodes de cette interface */

    private final MyGLRenderer mRenderer;

    Timer timer = new Timer();

    public MyGLSurfaceView(Context context, MyGLRenderer mRenderer, Timer timer) {
        super(context);
        this.mRenderer = mRenderer;

    }

    public MyGLSurfaceView(Context context, int nbRowGrid, int nbColumnGrid) {
        super(context);
        setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        // Création d'un context OpenGLES 3.0
        setEGLContextClientVersion(3);

        Log.d("MyGLSurfaceView", String.valueOf(nbRowGrid));
        Log.d("MyGLSurfaceView", String.valueOf(nbColumnGrid));

        // Création du renderer qui va être lié au conteneur View créé
        mRenderer = new MyGLRenderer(nbRowGrid, nbColumnGrid);
        setRenderer(mRenderer);



        // Option pour indiquer qu'on redessine uniquement si les données changent
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if(!mRenderer.isAtTheBottom()){
                    mRenderer.descendreBlock();
                }
                else{
                    mRenderer.createBlock();
                }
                requestRender();
            }
        }, 2000, 1000);
    }

    /* pour gérer la translation */
    private float mPreviousX;
    private float mPreviousY;
    private boolean condition = false;

    /* Comment interpréter les événements sur l'écran tactile */
    @Override
    public boolean onTouchEvent(MotionEvent e) {

        float x = e.getX();
        float y = e.getY();

        // la taille de l'écran en pixels
        float screen_x = getWidth();
        float screen_y = getHeight();

        /*
        Log.d(TAG + "message", "x"+Float.toString(x));
        Log.d(TAG + "message", "y"+Float.toString(y));
        Log.d(TAG + "message", "screen_x="+Float.toString(screen_x));
        Log.d(TAG + "message", "screen_y="+Float.toString(screen_y));


         */
        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                // Vérifiez si le toucher se trouve dans la zone de votre bloc
                /*
                if (x >= bloc.x && x < bloc.x + bloc.width && y >= bloc.y && y < bloc.y + bloc.height) {
                    // Le toucher est dans la zone de votre bloc, vous pouvez effectuer l'action souhaitée
                    // ...
                }

                 */

                //Log.d(TAG + " event", "DOWN");

                // Faire la rotate ici




                this.mRenderer.rotate();
                //requestRender();



                break;

            //case MotionEvent.ACTION

            case MotionEvent.ACTION_MOVE:
                // Effectuez l'action souhaitée lors du déplacement
                // ...

                //Log.d(TAG + " event", "MOVE");

                // Déplacer horizontalement le block ici
                /*
                System.out.println("x = " + x);
                System.out.println("get = " + e.getX());
                if(x + e.getX() < -200){
                    this.mRenderer.deplacerBlockGauche();
                    requestRender();
                }

                 */

                break;
        }

        return true;
    }

}
