package com.ai_glass.audio

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

/**
 * PhoneSpeechModule:
 * Native Android module providing speech recognition (ASR) via Android's SpeechRecognizer.
 * Uses device microphone and recognizes Vietnamese (vi-VN) directly.
 */
class PhoneSpeechModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val TAG = "PhoneSpeechModule"
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var activePromise: Promise? = null
    private var lastRecognizedText: String = ""
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun getName(): String = "PhoneSpeechModule"

    @ReactMethod
    fun isAvailable(promise: Promise) {
        mainHandler.post {
            try {
                val available = SpeechRecognizer.isRecognitionAvailable(reactContext)
                promise.resolve(available)
            } catch (e: Exception) {
                promise.resolve(false)
            }
        }
    }

    @ReactMethod
    fun startListening(promise: Promise) {
        mainHandler.post {
            try {
                if (isListening) {
                    stopInternal()
                }

                lastRecognizedText = ""
                activePromise = null

                if (speechRecognizer == null) {
                    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(reactContext)
                }

                speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        Log.d(TAG, "onReadyForSpeech")
                    }

                    override fun onBeginningOfSpeech() {
                        Log.d(TAG, "onBeginningOfSpeech")
                    }

                    override fun onRmsChanged(rmsdB: Float) {}

                    override fun onBufferReceived(buffer: ByteArray?) {}

                    override fun onEndOfSpeech() {
                        Log.d(TAG, "onEndOfSpeech")
                    }

                    override fun onError(error: Int) {
                        Log.w(TAG, "Speech recognition error code: $error")
                        val text = lastRecognizedText
                        val promiseToResolve = activePromise
                        stopInternal()
                        val result = Arguments.createMap().apply {
                            putString("text", text)
                            putInt("errorCode", error)
                        }
                        promiseToResolve?.resolve(result)
                    }

                    override fun onResults(results: Bundle?) {
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val text = if (!matches.isNullOrEmpty()) matches[0] else lastRecognizedText
                        Log.d(TAG, "onResults: $text")
                        isListening = false
                        val result = Arguments.createMap().apply {
                            putString("text", text)
                            putInt("errorCode", 0)
                        }
                        activePromise?.resolve(result)
                        activePromise = null
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!matches.isNullOrEmpty()) {
                            lastRecognizedText = matches[0]
                            Log.d(TAG, "onPartialResults: $lastRecognizedText")
                        }
                    }

                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "vi-VN")
                    putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, "vi-VN")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                }

                speechRecognizer?.startListening(intent)
                isListening = true
                Log.d(TAG, "SpeechRecognizer started listening (vi-VN)")
                promise.resolve(true)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start listening", e)
                isListening = false
                promise.reject("ERR_START", e.message)
            }
        }
    }

    @ReactMethod
    fun stopListening(promise: Promise) {
        mainHandler.post {
            try {
                if (isListening) {
                    activePromise = promise
                    speechRecognizer?.stopListening()
                    // Safety timeout if onResults doesn't fire promptly
                    mainHandler.postDelayed({
                        if (activePromise != null) {
                            val result = Arguments.createMap().apply {
                                putString("text", lastRecognizedText)
                                putInt("errorCode", 0)
                            }
                            activePromise?.resolve(result)
                            activePromise = null
                            isListening = false
                        }
                    }, 2000)
                } else {
                    val result = Arguments.createMap().apply {
                        putString("text", lastRecognizedText)
                        putInt("errorCode", 0)
                    }
                    promise.resolve(result)
                }
            } catch (e: Exception) {
                promise.reject("ERR_STOP", e.message)
            }
        }
    }

    @ReactMethod
    fun cancelListening(promise: Promise) {
        mainHandler.post {
            stopInternal()
            promise.resolve(true)
        }
    }

    private fun stopInternal() {
        try {
            isListening = false
            activePromise = null
            speechRecognizer?.cancel()
            speechRecognizer?.destroy()
            speechRecognizer = null
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping SpeechRecognizer", e)
        }
    }

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        mainHandler.post {
            stopInternal()
        }
    }
}
