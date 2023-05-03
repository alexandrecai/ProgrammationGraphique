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

    /* pour gérer la translation */
    private float mPreviousX;
    private float mPreviousY;
    private boolean mIsSwiping = false;


    private static final int SWIPE_THRESHOLD = 100;

    public static final String TAG = "MyGLSurfaceView";

    Timer timer = new Timer();


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
        this.mRenderer.setGrid();



        // Option pour indiquer qu'on redessine uniquement si les données changent
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if(!mRenderer.isAtTheBottom() && !mRenderer.collision()){
                    mRenderer.descendreBlock();
                }
                requestRender();
            }
        }, 2000, 1000);

    }



    /* Comment interpréter les événements sur l'écran tactile */
    @Override
    public boolean onTouchEvent(MotionEvent e) {

        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                mPreviousX = e.getX();
                mPreviousY = e.getY();
                mIsSwiping = false;
                return true;

            case MotionEvent.ACTION_MOVE:
                if (!mIsSwiping) {
                    float deltaX = e.getX() - mPreviousX;
                    float deltaY = e.getY() - mPreviousY;
                    if (Math.abs(deltaX) > SWIPE_THRESHOLD && Math.abs(deltaY) < SWIPE_THRESHOLD) {
                        mIsSwiping = true;
                        if (deltaX < 0) {
                            Log.d(TAG + "event", "SWIPE LEFT");
                            mRenderer.deplacerBlockGauche();

                        } else {
                            Log.d(TAG + "event", "SWIPE RIGHT");
                            mRenderer.deplacerBlockDroite();

                        }
                    }
                }
                return true;

            case MotionEvent.ACTION_UP:
                if (!mIsSwiping) {
                    Log.d(TAG + "event", "SINGLE TAP");
                    mRenderer.rotate();
                }
                return true;
        }
        return true;
    }
}
