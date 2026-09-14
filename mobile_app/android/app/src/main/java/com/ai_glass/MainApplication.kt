package com.ai_glass

import android.app.Application
import com.facebook.react.PackageList
import com.facebook.react.ReactApplication
import com.facebook.react.ReactHost
import com.facebook.react.ReactNativeApplicationEntryPoint.loadReactNative
import com.facebook.react.defaults.DefaultReactHost.getDefaultReactHost

class MainApplication : Application(), ReactApplication {

  override val reactHost: ReactHost by lazy {
    getDefaultReactHost(
      context = applicationContext,
      packageList =
        PackageList(this).packages.apply {
          // Register BSmart native ONNX Runtime module
          add(com.ai_glass.onnx.OnnxInferencePackage())
          // Register BSmart llama.cpp GGUF VLM module for opt-in Feature 1 experiments
          add(com.ai_glass.llama.LlamaCppVlmPackage())
          // Register BSmart native Audio Player module for activation voice clips
          add(com.ai_glass.audio.AudioPlayerPackage())
          // Register BSmart native Phone Camera module for standalone mode
          add(com.ai_glass.camera.PhoneCameraPackage())
          add(com.ai_glass.storage.ImageStoragePackage())
          add(com.ai_glass.location.LocationPackage())
          // Register BSmart Foreground Service module for background life-cycle protection
          add(com.ai_glass.service.ForegroundServicePackage())
        },
    )
  }

  override fun onCreate() {
    super.onCreate()
    loadReactNative(this)
  }
}
