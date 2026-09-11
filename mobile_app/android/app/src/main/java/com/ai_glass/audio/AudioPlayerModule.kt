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

    @ReactMethod
    fun playBeep(toneType: String, promise: Promise) {
        try {
            val toneGenerator = android.media.ToneGenerator(android.media.AudioManager.STREAM_MUSIC, 100)
            val tone = when (toneType.lowercase()) {
                "start", "beep" -> android.media.ToneGenerator.TONE_PROP_BEEP
                "stop", "release", "click" -> android.media.ToneGenerator.TONE_PROP_ACK
                "cancel", "error" -> android.media.ToneGenerator.TONE_PROP_NACK
                else -> android.media.ToneGenerator.TONE_PROP_BEEP
            }
            toneGenerator.startTone(tone, 150)
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                try {
                    toneGenerator.release()
                } catch (ignored: Exception) {}
            }, 250)
            promise.resolve(true)
        } catch (e: Exception) {
            android.util.Log.e("AudioPlayerModule", "Failed to play beep tone: $toneType", e)
            promise.reject("ERR_BEEP", e.message, e)
        }
    }

    private var nativeTts: android.speech.tts.TextToSpeech? = null
    private var ttsReady = false

    @ReactMethod
    fun synthesizeSpeechToPcm(text: String, promise: Promise) {
        if (text.isBlank()) {
            promise.resolve("")
            return
        }

        val runSynth = {
            try {
                val tempFile = java.io.File(reactContext.cacheDir, "tts_out_${System.currentTimeMillis()}.wav")
                val utteranceId = "synth_${System.currentTimeMillis()}"

                nativeTts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(id: String?) {}

                    override fun onDone(id: String?) {
                        if (id == utteranceId && tempFile.exists()) {
                            try {
                                val bytes = tempFile.readBytes()
                                tempFile.delete()
                                // Skip 44-byte WAV header if present to get pure PCM samples
                                val pcmBytes = if (bytes.size > 44 && bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte()) {
                                    bytes.copyOfRange(44, bytes.size)
                                } else {
                                    bytes
                                }
                                val base64 = android.util.Base64.encodeToString(pcmBytes, android.util.Base64.NO_WRAP)
                                promise.resolve(base64)
                            } catch (e: Exception) {
                                promise.reject("ERR_READ", e.message)
                            }
                        }
                    }

                    override fun onError(id: String?) {
                        if (id == utteranceId) {
                            tempFile.delete()
                            promise.reject("ERR_SYNTH", "TTS synthesis failed")
                        }
                    }
                })

                val params = android.os.Bundle()
                val result = nativeTts?.synthesizeToFile(text, params, tempFile, utteranceId)
                if (result != android.speech.tts.TextToSpeech.SUCCESS) {
                    tempFile.delete()
                    promise.reject("ERR_SYNTH_START", "Failed to start TTS synthesis to file")
                }
            } catch (e: Exception) {
                promise.reject("ERR_SYNTH_EXCEPTION", e.message)
            }
        }

        if (nativeTts == null || !ttsReady) {
            nativeTts = android.speech.tts.TextToSpeech(reactContext) { status ->
                if (status == android.speech.tts.TextToSpeech.SUCCESS) {
                    nativeTts?.language = java.util.Locale("vi", "VN")
                    ttsReady = true
                    runSynth()
                } else {
                    promise.reject("ERR_TTS_INIT", "Native TTS init failed")
                }
            }
        } else {
            runSynth()
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

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        releasePlayer()
        try {
            nativeTts?.stop()
            nativeTts?.shutdown()
            nativeTts = null
            ttsReady = false
        } catch (e: Exception) {
            Log.w("AudioPlayerModule", "Error shutting down native TTS", e)
        }
    }
}
