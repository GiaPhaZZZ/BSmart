package com.ai_glass.audio

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Log
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class AudioPlayerModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private var mediaPlayer: MediaPlayer? = null
    private val lock = Any()

    override fun getName(): String = "AudioPlayerModule"

    @ReactMethod
    fun isAvailable(promise: Promise) {
        promise.resolve(true)
    }

    @ReactMethod
    fun playSound(name: String, promise: Promise) {
        synchronized(lock) {
            try {
                releasePlayer()

                val cleanName = name.lowercase().replace("-", "_").trim()
                val resId = reactContext.resources.getIdentifier(
                    cleanName, "raw", reactContext.packageName
                )

                if (resId == 0) {
                    Log.w("AudioPlayerModule", "Sound resource not found: $cleanName")
                    promise.reject("ERR_NOT_FOUND", "Sound resource not found: $cleanName")
                    return
                }

                val player = MediaPlayer.create(reactContext, resId)
                if (player == null) {
                    Log.w("AudioPlayerModule", "Failed to create MediaPlayer for: $cleanName")
                    promise.reject("ERR_CREATE", "Failed to create MediaPlayer for $cleanName")
                    return
                }

                player.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .build()
                )

                var promiseSettled = false

                player.setOnCompletionListener { mp ->
                    synchronized(lock) {
                        try {
                            mp.release()
                            if (mediaPlayer === mp) {
                                mediaPlayer = null
                            }
                        } catch (e: Exception) {
                            Log.e("AudioPlayerModule", "Error releasing player on completion", e)
                        }
                    }
                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.resolve(true)
                    }
                }

                player.setOnErrorListener { mp, what, extra ->
                    synchronized(lock) {
                        try {
                            mp.release()
                            if (mediaPlayer === mp) {
                                mediaPlayer = null
                            }
                        } catch (e: Exception) {
                            Log.e("AudioPlayerModule", "Error releasing player on error", e)
                        }
                    }
                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.reject("ERR_PLAYBACK", "Playback error: what=$what, extra=$extra")
                    }
                    true
                }

                mediaPlayer = player
                player.start()
                Log.i("AudioPlayerModule", "Started playing sound: $cleanName")
            } catch (e: Exception) {
                Log.e("AudioPlayerModule", "Exception playing sound: $name", e)
                promise.reject("ERR_EXCEPTION", e.message, e)
            }
        }
    }

    @ReactMethod
    fun stopSound(promise: Promise) {
        synchronized(lock) {
            try {
                releasePlayer()
                promise.resolve(true)
            } catch (e: Exception) {
                promise.reject("ERR_STOP", e.message, e)
            }
        }
    }

    private fun releasePlayer() {
        mediaPlayer?.let {
            try {
                if (it.isPlaying) {
                    it.stop()
                }
                it.release()
            } catch (e: Exception) {
                Log.w("AudioPlayerModule", "Error releasing player", e)
            }
            mediaPlayer = null
        }
    }
}
