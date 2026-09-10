package com.ai_glass.service

import android.content.Intent
import android.os.Build
import android.util.Log
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class ForegroundServiceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "ForegroundServiceModule"

    @ReactMethod
    fun startService(promise: Promise) {
        try {
            val intent = Intent(reactContext, BSmartForegroundService::class.java).apply {
                action = BSmartForegroundService.ACTION_START
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                reactContext.startForegroundService(intent)
            } else {
                reactContext.startService(intent)
            }
            Log.i("ForegroundService", "Foreground service started successfully")
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e("ForegroundService", "Failed to start foreground service", e)
            promise.reject("ERR_SERVICE_START", e.message, e)
        }
    }

    @ReactMethod
    fun stopService(promise: Promise) {
        try {
            val intent = Intent(reactContext, BSmartForegroundService::class.java).apply {
                action = BSmartForegroundService.ACTION_STOP
            }
            reactContext.startService(intent)
            Log.i("ForegroundService", "Foreground service stopped")
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e("ForegroundService", "Failed to stop foreground service", e)
            promise.reject("ERR_SERVICE_STOP", e.message, e)
        }
    }
}
