package com.ai_glass.llama

import android.os.SystemClock
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.ceil

object LlamaCppVlmNative {
    val loadError: String?

    init {
        var err: String? = null
        try {
            System.loadLibrary("bsmart_llama_vlm")
        } catch (throwable: Throwable) {
            err = throwable.message ?: throwable.javaClass.simpleName
        }
        loadError = err
    }

    external fun getStatusNative(): String
    external fun loadModelNative(
        modelPath: String,
        mmprojPath: String,
        contextSize: Int,
        threads: Int,
        maxTokens: Int,
        temperature: Double
    ): String
    external fun runVisualQANative(
        imagePath: String,
        prompt: String,
        maxTokens: Int,
        temperature: Double
    ): String
    external fun unloadModelNative(): String
    external fun setImageMaxTokensNative(tokens: Int)
}

class LlamaCppVlmModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val tag = "LlamaCppVlmModule"

    override fun getName(): String = "LlamaCppVlmModule"

    private fun ggufDir(): File = File(reactContext.filesDir, "gguf").apply { mkdirs() }

    private fun defaultModelPath(): String =
        File(ggufDir(), "SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf").absolutePath

    private fun defaultMmprojPath(): String =
        File(ggufDir(), "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf").absolutePath

    private fun optionString(options: ReadableMap, key: String, fallback: String): String =
        if (options.hasKey(key) && !options.isNull(key)) {
            options.getString(key)?.takeIf { it.isNotBlank() } ?: fallback
        } else {
            fallback
        }

    private fun optionInt(options: ReadableMap, key: String, fallback: Int): Int =
        if (options.hasKey(key) && !options.isNull(key)) options.getInt(key) else fallback

    private fun optionDouble(options: ReadableMap, key: String, fallback: Double): Double =
        if (options.hasKey(key) && !options.isNull(key)) options.getDouble(key) else fallback

    private fun jsonToMap(json: String): WritableMap {
        val parsed = JSONObject(json)
        return objectToMap(parsed)
    }

    private fun objectToMap(obj: JSONObject): WritableMap {
        val map = Arguments.createMap()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            when (val value = obj.get(key)) {
                JSONObject.NULL -> map.putNull(key)
                is JSONObject -> map.putMap(key, objectToMap(value))
                is JSONArray -> map.putArray(key, arrayToWritableArray(value))
                is Boolean -> map.putBoolean(key, value)
                is Int -> map.putInt(key, value)
                is Long -> map.putDouble(key, value.toDouble())
                is Double -> map.putDouble(key, value)
                is Float -> map.putDouble(key, value.toDouble())
                else -> map.putString(key, value.toString())
            }
        }
        return map
    }

    private fun arrayToWritableArray(array: JSONArray): WritableArray {
        val out = Arguments.createArray()
        for (i in 0 until array.length()) {
            when (val value = array.get(i)) {
                JSONObject.NULL -> out.pushNull()
                is JSONObject -> out.pushMap(objectToMap(value))
                is JSONArray -> out.pushArray(arrayToWritableArray(value))
                is Boolean -> out.pushBoolean(value)
                is Int -> out.pushInt(value)
                is Long -> out.pushDouble(value.toDouble())
                is Double -> out.pushDouble(value)
                is Float -> out.pushDouble(value.toDouble())
                else -> out.pushString(value.toString())
            }
        }
        return out
    }

    private fun unavailableMap(): WritableMap =
        Arguments.createMap().apply {
            putBoolean("nativeLibraryLoaded", false)
            putBoolean("runtimeAvailable", false)
            putBoolean("compiledWithLlamaCpp", false)
            putBoolean("modelLoaded", false)
            putString("runtime", "unavailable")
            putString("message", LlamaCppVlmNative.loadError ?: "bsmart_llama_vlm could not be loaded")
            putString("defaultModelPath", defaultModelPath())
            putString("defaultMmprojPath", defaultMmprojPath())
            putBoolean("defaultModelExists", File(defaultModelPath()).exists())
            putBoolean("defaultMmprojExists", File(defaultMmprojPath()).exists())
        }

    private fun rejectNativeUnavailable(promise: Promise) {
        promise.reject(
            "NATIVE_LIBRARY_UNAVAILABLE",
            LlamaCppVlmNative.loadError ?: "bsmart_llama_vlm native library is unavailable"
        )
    }

    private fun withNativeResult(promise: Promise, block: () -> String) {
        if (LlamaCppVlmNative.loadError != null) {
            rejectNativeUnavailable(promise)
            return
        }
        try {
            promise.resolve(jsonToMap(block()))
        } catch (e: Exception) {
            promise.reject("LLAMA_CPP_VLM_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun getStatus(promise: Promise) {
        if (LlamaCppVlmNative.loadError != null) {
            promise.resolve(unavailableMap())
            return
        }
        withNativeResult(promise) {
            LlamaCppVlmNative.getStatusNative()
        }
    }

    @ReactMethod
    fun getDefaultModelPaths(promise: Promise) {
        promise.resolve(
            Arguments.createMap().apply {
                putString("modelPath", defaultModelPath())
                putString("mmprojPath", defaultMmprojPath())
                putBoolean("modelExists", File(defaultModelPath()).exists())
                putBoolean("mmprojExists", File(defaultMmprojPath()).exists())
            }
        )
    }

    @ReactMethod
    fun loadModel(modelPath: String, mmprojPath: String, options: ReadableMap, promise: Promise) {
        val resolvedModelPath = modelPath.takeIf { it.isNotBlank() }
            ?: optionString(options, "modelPath", defaultModelPath())
        val resolvedMmprojPath = mmprojPath.takeIf { it.isNotBlank() }
            ?: optionString(options, "mmprojPath", defaultMmprojPath())
        val contextSize = optionInt(options, "contextSize", 2048)
        val threads = optionInt(options, "threads", 6)
        val maxTokens = optionInt(options, "maxTokens", 16)
        val imageMaxTokens = optionInt(options, "imageMaxTokens", 64)
        val temperature = optionDouble(options, "temperature", 0.0)

        LlamaCppVlmNative.setImageMaxTokensNative(imageMaxTokens)

        withNativeResult(promise) {
            LlamaCppVlmNative.loadModelNative(
                resolvedModelPath,
                resolvedMmprojPath,
                contextSize,
                threads,
                maxTokens,
                temperature
            )
        }
    }

    @ReactMethod
    fun runVisualQA(imagePath: String, prompt: String, options: ReadableMap, promise: Promise) {
        val maxTokens = optionInt(options, "maxTokens", 48)
        val temperature = optionDouble(options, "temperature", 0.0)
        withNativeResult(promise) {
            LlamaCppVlmNative.runVisualQANative(imagePath, prompt, maxTokens, temperature)
        }
    }

    @ReactMethod
    fun runVisualQABase64(base64Image: String, prompt: String, options: ReadableMap, promise: Promise) {
        if (base64Image.isBlank()) {
            promise.resolve(
                Arguments.createMap().apply {
                    putBoolean("isSuccess", false)
                    putString("error", "INVALID_IMAGE")
                    putString("message", "Image base64 is empty")
                }
            )
            return
        }

        Thread {
            var tempFile: File? = null
            try {
                val cleanBase64 = if (base64Image.contains(",")) {
                    base64Image.substringAfter(",")
                } else {
                    base64Image
                }
                val decoded = Base64.decode(cleanBase64, Base64.DEFAULT)
                tempFile = File(reactContext.cacheDir, "bsmart_vlm_${SystemClock.elapsedRealtimeNanos()}.jpg")
                tempFile.writeBytes(decoded)

                val maxTokens = optionInt(options, "maxTokens", 48)
                val temperature = optionDouble(options, "temperature", 0.0)
                val json = if (LlamaCppVlmNative.loadError == null) {
                    LlamaCppVlmNative.runVisualQANative(tempFile.absolutePath, prompt, maxTokens, temperature)
                } else {
                    throw IllegalStateException(LlamaCppVlmNative.loadError)
                }
                promise.resolve(jsonToMap(json))
            } catch (e: Exception) {
                android.util.Log.e(tag, "runVisualQABase64 failed", e)
                promise.reject("LLAMA_CPP_VLM_ERROR", e.message, e)
            } finally {
                tempFile?.delete()
            }
        }.start()
    }

    @ReactMethod
    fun benchmarkVisualQABase64(
        base64Image: String,
        prompts: ReadableArray,
        runs: Int,
        options: ReadableMap,
        promise: Promise
    ) {
        Thread {
            var tempFile: File? = null
            try {
                val cleanBase64 = if (base64Image.contains(",")) {
                    base64Image.substringAfter(",")
                } else {
                    base64Image
                }
                val decoded = Base64.decode(cleanBase64, Base64.DEFAULT)
                tempFile = File(reactContext.cacheDir, "bsmart_vlm_bench_${SystemClock.elapsedRealtimeNanos()}.jpg")
                tempFile.writeBytes(decoded)

                val measuredRuns = runs.coerceAtLeast(1)
                val promptCount = prompts.size()
                val maxTokens = optionInt(options, "maxTokens", 48)
                val temperature = optionDouble(options, "temperature", 0.0)
                val results = Arguments.createArray()
                val totals = mutableListOf<Double>()

                repeat(measuredRuns) { index ->
                    val prompt = if (promptCount > 0) {
                        prompts.getString(index % promptCount) ?: "Briefly describe the scene."
                    } else {
                        "Briefly describe the scene."
                    }
                    val json = LlamaCppVlmNative.runVisualQANative(
                        tempFile.absolutePath,
                        prompt,
                        maxTokens,
                        temperature
                    )
                    val map = jsonToMap(json)
                    val timing = map.getMap("timings")
                    val total = timing?.getDouble("total_vlm_ms") ?: Double.NaN
                    if (!total.isNaN()) totals += total
                    results.pushMap(map)
                }

                val sorted = totals.sorted()
                val summary = Arguments.createMap().apply {
                    putInt("runs", measuredRuns)
                    putDouble("min_ms", sorted.firstOrNull() ?: Double.NaN)
                    putDouble("max_ms", sorted.lastOrNull() ?: Double.NaN)
                    putDouble("average_ms", if (totals.isNotEmpty()) totals.average() else Double.NaN)
                    putDouble("median_ms", percentile(sorted, 0.50))
                    putDouble("p90_ms", percentile(sorted, 0.90))
                }

                promise.resolve(
                    Arguments.createMap().apply {
                        putArray("results", results)
                        putMap("summary", summary)
                    }
                )
            } catch (e: Exception) {
                android.util.Log.e(tag, "benchmarkVisualQABase64 failed", e)
                promise.reject("LLAMA_CPP_VLM_BENCH_ERROR", e.message, e)
            } finally {
                tempFile?.delete()
            }
        }.start()
    }

    @ReactMethod
    fun unloadModel(promise: Promise) {
        withNativeResult(promise) {
            LlamaCppVlmNative.unloadModelNative()
        }
    }

    private fun percentile(values: List<Double>, p: Double): Double {
        if (values.isEmpty()) return Double.NaN
        if (values.size == 1) return values.first()
        val rank = (values.size - 1) * p
        val low = rank.toInt()
        val high = ceil(rank).toInt()
        if (low == high) return values[low]
        val fraction = rank - low
        return values[low] * (1.0 - fraction) + values[high] * fraction
    }
}
