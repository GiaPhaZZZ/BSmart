package com.ai_glass.onnx

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import com.facebook.react.bridge.*
import org.json.JSONArray
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

class OnnxInferenceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val tag = "OnnxInferenceModule"
    private val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = ConcurrentHashMap<String, OrtSession>()
    private val cocoLabelsVi = ConcurrentHashMap<Int, String>()
    private val modelCopyBufferSize = 64 * 1024
    private val objectDetectionModelIds = setOf("yolo26s", "zipdepth")
    private val vlmModelIds = setOf("smolvlm2_vis_hf", "smolvlm2_emb", "smolvlm2_dec")
    private val asrModelIds = setOf("phowhisper_encoder", "phowhisper_decoder")

    override fun getName(): String = "OnnxInferenceModule"

    init {
        loadCocoLabels()
    }

    private fun loadCocoLabels() {
        try {
            val assetManager = reactContext.assets
            val inputStream = assetManager.open("models/navigation/coco_labels_vi.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val jsonArray = JSONArray(jsonString)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val id = obj.getInt("id")
                val viName = obj.getString("vi")
                cocoLabelsVi[id] = viName
            }
        } catch (e: Exception) {
            android.util.Log.e(tag, "Failed to load coco_labels_vi.json", e)
            cocoLabelsVi[0] = "người"
            cocoLabelsVi[2] = "ô tô"
            cocoLabelsVi[3] = "xe máy"
            cocoLabelsVi[56] = "cái ghế"
        }
    }

    private fun logRuntimeMemory(label: String) {
        val runtime = Runtime.getRuntime()
        android.util.Log.i(
            tag,
            "$label Runtime totalMemory=${runtime.totalMemory()} freeMemory=${runtime.freeMemory()} maxMemory=${runtime.maxMemory()}"
        )
    }

    private fun logAssetBytes(label: String, assetPath: String): Long {
        val startedAt = System.nanoTime()
        var totalBytes = 0L
        try {
            reactContext.assets.open(assetPath).use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    totalBytes += read.toLong()
                }
            }
            android.util.Log.i(
                tag,
                "$label asset bytes=$totalBytes elapsedMs=${elapsedMs(startedAt)}"
            )
            return totalBytes
        } catch (e: Exception) {
            android.util.Log.e(tag, "$label asset open failed path=$assetPath elapsedMs=${elapsedMs(startedAt)}", e)
            throw e
        }
    }

    private fun elapsedMs(startedAt: Long): Long =
        (System.nanoTime() - startedAt) / 1_000_000L

    private fun shapeToString(shape: LongArray): String =
        shape.joinToString(prefix = "[", postfix = "]")

    private fun tensorShape(tensor: OnnxTensor): String =
        try {
            shapeToString(tensor.info.shape)
        } catch (_: Exception) {
            "[unknown]"
        }

    private fun shape3D(value: Array<Array<FloatArray>>): String =
        "[${value.size},${value.firstOrNull()?.size ?: 0},${value.firstOrNull()?.firstOrNull()?.size ?: 0}]"

    private fun shape2D(value: Array<FloatArray>): String =
        "[${value.size},${value.firstOrNull()?.size ?: 0}]"

    private fun logStageStart(stage: String, details: String = ""): Long {
        android.util.Log.i(tag, "$stage status=start elapsedMs=0 $details".trim())
        return System.nanoTime()
    }

    private fun logStageSuccess(stage: String, startedAt: Long, details: String = "") {
        android.util.Log.i(tag, "$stage status=success elapsedMs=${elapsedMs(startedAt)} $details".trim())
    }

    private fun logStageFailure(stage: String, startedAt: Long, details: String = "", e: Throwable) {
        android.util.Log.e(tag, "$stage status=failure elapsedMs=${elapsedMs(startedAt)} $details".trim(), e)
    }

    private fun extractedModelsDir(): File =
        File(reactContext.filesDir, "onnx_models").apply { mkdirs() }

    private fun deterministicModelFilename(assetPath: String): String =
        assetPath.substringAfterLast('/').replace(Regex("[^A-Za-z0-9._-]"), "_")

    private fun assetSize(assetPath: String): Long {
        return try {
            reactContext.assets.openFd(assetPath).use { it.length }
        } catch (_: Exception) {
            var totalBytes = 0L
            reactContext.assets.open(assetPath).use { input ->
                val buffer = ByteArray(modelCopyBufferSize)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    totalBytes += read.toLong()
                }
            }
            totalBytes
        }
    }

    private fun extractModelAssetIfNeeded(assetPath: String): File {
        val expectedSize = assetSize(assetPath)
        val outputFile = File(extractedModelsDir(), deterministicModelFilename(assetPath))
        if (outputFile.exists() && outputFile.length() == expectedSize) {
            return outputFile
        }

        val tempFile = File(outputFile.parentFile, "${outputFile.name}.tmp")
        reactContext.assets.open(assetPath).use { input ->
            tempFile.outputStream().use { output ->
                val buffer = ByteArray(modelCopyBufferSize)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    output.write(buffer, 0, read)
                }
            }
        }

        if (tempFile.length() != expectedSize) {
            tempFile.delete()
            throw IllegalStateException(
                "Extracted model size mismatch for $assetPath: got ${tempFile.length()}, expected $expectedSize"
            )
        }

        if (outputFile.exists() && !outputFile.delete()) {
            tempFile.delete()
            throw IllegalStateException("Could not replace cached model file ${outputFile.absolutePath}")
        }
        if (!tempFile.renameTo(outputFile)) {
            tempFile.delete()
            throw IllegalStateException("Could not move cached model file to ${outputFile.absolutePath}")
        }
        return outputFile
    }

    private fun createSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
            setMemoryPatternOptimization(false)
            setCPUArenaAllocator(false)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        }

    private fun isMemoryTight(): Boolean {
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        val availableUntilMax = runtime.maxMemory() - used
        return availableUntilMax < 128L * 1024L * 1024L
    }

    private fun releaseSessions(modelIds: Set<String>) {
        modelIds.forEach { modelId ->
            try {
                sessions.remove(modelId)?.close()
            } catch (e: Exception) {
                android.util.Log.w(tag, "Failed to close session $modelId", e)
            }
        }
    }

    private fun releaseObjectDetectionModelsInternal() = releaseSessions(objectDetectionModelIds)

    private fun releaseVlmModelsInternal() = releaseSessions(vlmModelIds)

    private fun releaseAsrModelsInternal() = releaseSessions(asrModelIds)

    private fun autoLoadModel(modelId: String, assetPath: String): Boolean {
        if (sessions.containsKey(modelId)) return true
        val startedAt = System.nanoTime()
        return try {
            logRuntimeMemory("before loading ${assetPath.substringAfterLast('/')}")
            val modelFile = extractModelAssetIfNeeded(assetPath)
            createSessionOptions().use { sessionOptions ->
                sessions[modelId] = environment.createSession(modelFile.absolutePath, sessionOptions)
            }
            android.util.Log.i(
                tag,
                "model_load status=success model=${assetPath.substringAfterLast('/')} asset=$assetPath file=${modelFile.absolutePath} bytes=${modelFile.length()} elapsedMs=${elapsedMs(startedAt)}"
            )
            true
        } catch (e: Exception) {
            android.util.Log.e(
                tag,
                "model_load status=failure model=${assetPath.substringAfterLast('/')} asset=$assetPath elapsedMs=${elapsedMs(startedAt)}",
                e
            )
            false
        }
    }

    @ReactMethod
    fun isNativeRuntimeAvailable(promise: Promise) {
        try {
            promise.resolve(environment != null)
        } catch (e: Exception) {
            promise.resolve(false)
        }
    }

    @ReactMethod
    fun loadModel(modelId: String, assetPath: String, promise: Promise) {
        if (autoLoadModel(modelId, assetPath)) {
            promise.resolve(true)
        } else {
            promise.reject("MODEL_LOAD_ERROR", "Failed to load $modelId from $assetPath")
        }
    }

    @ReactMethod
    fun releaseObjectDetectionModels(promise: Promise) {
        releaseObjectDetectionModelsInternal()
        promise.resolve(true)
    }

    @ReactMethod
    fun releaseVlmModels(promise: Promise) {
        releaseVlmModelsInternal()
        promise.resolve(true)
    }

    @ReactMethod
    fun releaseAsrModels(promise: Promise) {
        releaseAsrModelsInternal()
        promise.resolve(true)
    }

    data class BBox(val classId: Int, val score: Float, val cx: Float, val cy: Float, val w: Float, val h: Float) {
        val x1 get() = cx - w / 2
        val y1 get() = cy - h / 2
        val x2 get() = cx + w / 2
        val y2 get() = cy + h / 2
        val area get() = w * h
    }

    private fun computeIoU(box1: BBox, box2: BBox): Float {
        val interX1 = max(box1.x1, box2.x1)
        val interY1 = max(box1.y1, box2.y1)
        val interX2 = min(box1.x2, box2.x2)
        val interY2 = min(box1.y2, box2.y2)
        val interArea = max(0f, interX2 - interX1) * max(0f, interY2 - interY1)
        if (interArea == 0f) return 0f
        val unionArea = box1.area + box2.area - interArea
        return interArea / unionArea
    }

    @ReactMethod
    fun runObjectDetection(base64Image: String, promise: Promise) {
        Thread {
            var bitmap: Bitmap? = null
            var resized: Bitmap? = null
            var inputTensor: OnnxTensor? = null
            try {
                if (isMemoryTight()) {
                    releaseVlmModelsInternal()
                    releaseAsrModelsInternal()
                }
                autoLoadModel("yolo26s", "models/navigation/yolo26s.onnx")
                val session = sessions["yolo26s"]
                if (session == null) {
                    promise.reject("MODEL_NOT_LOADED", "yolo26s model not loaded")
                    return@Thread
                }
                val decodedBytes = Base64.decode(base64Image, Base64.DEFAULT)
                bitmap = BitmapFactory.decodeStream(ByteArrayInputStream(decodedBytes))
                if (bitmap == null) {
                    promise.reject("INVALID_IMAGE", "Could not decode base64 into Bitmap")
                    return@Thread
                }
                resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
                val inputBuffer = FloatBuffer.allocate(1 * 3 * 640 * 640)
                val pixels = IntArray(640 * 640)
                resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
                val rOffset = 0
                val gOffset = 640 * 640
                val bOffset = 2 * 640 * 640
                for (i in pixels.indices) {
                    val p = pixels[i]
                    inputBuffer.put(rOffset + i, ((p shr 16) and 0xFF) / 255.0f)
                    inputBuffer.put(gOffset + i, ((p shr 8) and 0xFF) / 255.0f)
                    inputBuffer.put(bOffset + i, (p and 0xFF) / 255.0f)
                }
                inputTensor = OnnxTensor.createTensor(environment, inputBuffer, longArrayOf(1, 3, 640, 640))
                val inputName = session.inputNames.iterator().next()
                val output = session.run(mapOf(inputName to inputTensor))
                val outputTensor = output[0].value as Array<Array<FloatArray>>
                val data = outputTensor[0]
                val boxes = mutableListOf<BBox>()
                val numClasses = 80
                val numAnchors = 8400
                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    var maxClassId = -1
                    for (c in 0 until numClasses) {
                        val score = data[4 + c][i]
                        if (score > maxScore) {
                            maxScore = score
                            maxClassId = c
                        }
                    }
                    if (maxScore > 0.5f) {
                        boxes.add(BBox(maxClassId, maxScore, data[0][i] / 640f, data[1][i] / 640f, data[2][i] / 640f, data[3][i] / 640f))
                    }
                }
                output.close()
                boxes.sortByDescending { it.score }
                val selectedBoxes = mutableListOf<BBox>()
                for (box in boxes) {
                    var suppress = false
                    for (selBox in selectedBoxes) {
                        if (box.classId == selBox.classId && computeIoU(box, selBox) > 0.45f) {
                            suppress = true
                            break
                        }
                    }
                    if (!suppress) selectedBoxes.add(box)
                }
                val resultsArray = Arguments.createArray()
                for (box in selectedBoxes) {
                    val map = Arguments.createMap()
                    map.putString("class", cocoLabelsVi[box.classId] ?: "vật thể")
                    map.putDouble("confidence", box.score.toDouble())
                    map.putDouble("x", box.cx.toDouble())
                    map.putDouble("y", box.cy.toDouble())
                    map.putDouble("width", box.w.toDouble())
                    map.putDouble("height", box.h.toDouble())
                    map.putDouble("depthScore", 0.5)
                    resultsArray.pushMap(map)
                }
                promise.resolve(resultsArray)
            } catch (e: Exception) {
                promise.reject("INFERENCE_ERROR", "YOLO on-device inference failed: ${e.message}", e)
            } finally {
                inputTensor?.close()
                bitmap?.recycle()
                resized?.recycle()
            }
        }.start()
    }

    @ReactMethod
    fun runDepthEstimation(base64Image: String, promise: Promise) {
        Thread {
            var bitmap: Bitmap? = null
            var resized: Bitmap? = null
            var inputTensor: OnnxTensor? = null
            try {
                if (isMemoryTight()) {
                    releaseVlmModelsInternal()
                    releaseAsrModelsInternal()
                }
                autoLoadModel("zipdepth", "models/navigation/zipdepth.onnx")
                val session = sessions["zipdepth"]
                if (session == null) {
                    promise.reject("MODEL_NOT_LOADED", "zipdepth model not loaded")
                    return@Thread
                }
                val decodedBytes = Base64.decode(base64Image, Base64.DEFAULT)
                bitmap = BitmapFactory.decodeStream(ByteArrayInputStream(decodedBytes))
                if (bitmap == null) {
                    promise.reject("INVALID_IMAGE", "Could not decode base64 into Bitmap")
                    return@Thread
                }
                resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true)
                val inputBuffer = FloatBuffer.allocate(1 * 3 * 384 * 384)
                val pixels = IntArray(384 * 384)
                resized.getPixels(pixels, 0, 384, 0, 0, 384, 384)
                val rOffset = 0
                val gOffset = 384 * 384
                val bOffset = 2 * 384 * 384
                for (i in pixels.indices) {
                    val p = pixels[i]
                    inputBuffer.put(rOffset + i, ((p shr 16) and 0xFF) / 255.0f)
                    inputBuffer.put(gOffset + i, ((p shr 8) and 0xFF) / 255.0f)
                    inputBuffer.put(bOffset + i, (p and 0xFF) / 255.0f)
                }
                inputTensor = OnnxTensor.createTensor(environment, inputBuffer, longArrayOf(1, 3, 384, 384))
                val inputName = session.inputNames.iterator().next()
                val output = session.run(mapOf(inputName to inputTensor))
                val outputTensor = output[0].value as Array<Array<Array<FloatArray>>>
                val depthMap = outputTensor[0][0]
                var sum = 0f
                for (r in 0 until 384) {
                    for (c in 0 until 384) {
                        sum += depthMap[r][c]
                    }
                }
                output.close()
                val map = Arguments.createMap().apply {
                    putBoolean("success", true)
                    putDouble("relativeDepthMean", (sum / (384 * 384)).toDouble())
                }
                promise.resolve(map)
            } catch (e: Exception) {
                promise.reject("DEPTH_ERROR", "ZipDepth inference failed: ${e.message}", e)
            } finally {
                inputTensor?.close()
                bitmap?.recycle()
                resized?.recycle()
            }
        }.start()
    }

    internal data class SmolVLM2ImageInputs(
        val pixelValues: FloatBuffer,
        val pixelAttentionMask: Array<Array<Array<BooleanArray>>>,
        val numImages: Int,
        val rows: Int,
        val cols: Int
    )

    internal fun resizeKeepingAspect(bitmap: Bitmap, longestEdge: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val scale = longestEdge.toFloat() / max(width, height).toFloat()
        val resizedWidth = max(1, kotlin.math.round(width * scale).toInt())
        val resizedHeight = max(1, kotlin.math.round(height * scale).toInt())
        return Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
    }

    internal fun resizeForVisionEncoder(bitmap: Bitmap, maxImageSize: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val aspectRatio = width.toDouble() / height.toDouble()
        val resizedWidth: Int
        val resizedHeight: Int
        if (width >= height) {
            resizedWidth = ceil(width.toDouble() / maxImageSize).toInt() * maxImageSize
            val aspectHeight = (resizedWidth / aspectRatio).toInt()
            resizedHeight = ceil(aspectHeight.toDouble() / maxImageSize).toInt() * maxImageSize
        } else {
            resizedHeight = ceil(height.toDouble() / maxImageSize).toInt() * maxImageSize
            val aspectWidth = (resizedHeight * aspectRatio).toInt()
            resizedWidth = ceil(aspectWidth.toDouble() / maxImageSize).toInt() * maxImageSize
        }
        return Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
    }

    internal fun buildSmolVLM2ImageInputs(bitmap: Bitmap): SmolVLM2ImageInputs {
        val maxImageSize = 512
        val resizedToPromptSize = resizeKeepingAspect(bitmap, 2048)
        val visionSized = resizeForVisionEncoder(resizedToPromptSize, maxImageSize)
        if (resizedToPromptSize !== bitmap) resizedToPromptSize.recycle()

        val frames = mutableListOf<Bitmap>()
        var rows = 0
        var cols = 0
        if (visionSized.height > maxImageSize || visionSized.width > maxImageSize) {
            rows = ceil(visionSized.height.toDouble() / maxImageSize).toInt()
            cols = ceil(visionSized.width.toDouble() / maxImageSize).toInt()
            val optimalHeight = ceil(visionSized.height.toDouble() / rows).toInt()
            val optimalWidth = ceil(visionSized.width.toDouble() / cols).toInt()

            for (row in 0 until rows) {
                for (col in 0 until cols) {
                    val startX = col * optimalWidth
                    val startY = row * optimalHeight
                    val endX = min(startX + optimalWidth, visionSized.width)
                    val endY = min(startY + optimalHeight, visionSized.height)
                    val crop = Bitmap.createBitmap(visionSized, startX, startY, endX - startX, endY - startY)
                    val frame = if (crop.width == maxImageSize && crop.height == maxImageSize) {
                        crop
                    } else {
                        val scaled = Bitmap.createScaledBitmap(crop, maxImageSize, maxImageSize, true)
                        crop.recycle()
                        scaled
                    }
                    frames.add(frame)
                }
            }
            frames.add(Bitmap.createScaledBitmap(visionSized, maxImageSize, maxImageSize, true))
        } else {
            frames.add(visionSized)
        }
        if (!frames.contains(visionSized)) visionSized.recycle()

        val numImages = frames.size
        val plane = maxImageSize * maxImageSize
        val pixelValues = FloatBuffer.allocate(numImages * 3 * plane)
        val pixels = IntArray(plane)
        frames.forEachIndexed { imageIndex, frame ->
            frame.getPixels(pixels, 0, maxImageSize, 0, 0, maxImageSize, maxImageSize)
            val imageOffset = imageIndex * 3 * plane
            for (i in pixels.indices) {
                val p = pixels[i]
                pixelValues.put(imageOffset + i, (((p shr 16) and 0xFF) / 255.0f - 0.5f) / 0.5f)
                pixelValues.put(imageOffset + plane + i, (((p shr 8) and 0xFF) / 255.0f - 0.5f) / 0.5f)
                pixelValues.put(imageOffset + (2 * plane) + i, ((p and 0xFF) / 255.0f - 0.5f) / 0.5f)
            }
        }
        pixelValues.rewind()
        frames.forEach { it.recycle() }

        val pixelAttentionMask = Array(1) {
            Array(numImages) {
                Array(maxImageSize) {
                    BooleanArray(maxImageSize) { true }
                }
            }
        }
        return SmolVLM2ImageInputs(pixelValues, pixelAttentionMask, numImages, rows, cols)
    }

    internal fun buildSmolVLM2ExpandedPrompt(question: String, rows: Int, cols: Int): String {
        val imageSeqLen = 64
        val fake = "<fake_token_around_image>"
        val image = "<image>"
        val expandedImage = if (rows == 0 && cols == 0) {
            fake + "<global-img>" + image.repeat(imageSeqLen) + fake
        } else {
            val builder = StringBuilder()
            for (row in 1..rows) {
                for (col in 1..cols) {
                    builder.append(fake)
                    builder.append("<row_${row}_col_${col}>")
                    builder.append(image.repeat(imageSeqLen))
                }
                builder.append('\n')
            }
            builder.append('\n')
            builder.append(fake)
            builder.append("<global-img>")
            builder.append(image.repeat(imageSeqLen))
            builder.append(fake)
            builder.toString()
        }
        return "<|im_start|>User:${expandedImage}<end_of_utterance>\nUser: ${question}<end_of_utterance>\nAssistant:"
    }

    internal fun mergeSmolVLM2Embeddings(
        inputIds: IntArray,
        tokenEmbeds: Array<Array<FloatArray>>,
        imageFeats: Array<Array<FloatArray>>
    ): FloatArray {
        val embedDim = tokenEmbeds[0][0].size
        val imageSeqLen = imageFeats[0].size
        val imageTokenCount = inputIds.count { it == SmolVLM2Tokenizer.IMAGE_TOKEN_ID }
        val imageFeatureCount = imageFeats.size * imageSeqLen
        require(imageTokenCount == imageFeatureCount) {
            "Image token count mismatch: got $imageTokenCount, expected $imageFeatureCount"
        }

        val mergedFlat = FloatArray(inputIds.size * embedDim)
        var imagePositionIndex = 0
        for (tokenIndex in inputIds.indices) {
            if (inputIds[tokenIndex] == SmolVLM2Tokenizer.IMAGE_TOKEN_ID) {
                val blockIndex = imagePositionIndex / imageSeqLen
                val localIndex = imagePositionIndex % imageSeqLen
                for (d in 0 until embedDim) {
                    mergedFlat[tokenIndex * embedDim + d] = imageFeats[blockIndex][localIndex][d]
                }
                imagePositionIndex++
            } else {
                for (d in 0 until embedDim) {
                    mergedFlat[tokenIndex * embedDim + d] = tokenEmbeds[0][tokenIndex][d]
                }
            }
        }
        return mergedFlat
    }

    @ReactMethod
    fun runVisualQA(base64Image: String, question: String, promise: Promise) {
        // HF SmolVLM2 ONNX pipeline: split image tiles, expand image tokens,
        // replace image-token embeddings with vision features, then decode with KV cache.
        Thread {
            val tensorsToClose = mutableListOf<OnnxTensor>()
            val resultsToClose = mutableListOf<OrtSession.Result>()
            var bitmap: Bitmap? = null
            var currentStage = "VLM_STAGE_INIT"
            var currentStageStartedAt = System.nanoTime()
            var lastOnnxModel = "none"
            var lastOnnxInputs = "none"
            var lastOnnxOutputs = "none"

            fun beginStage(stage: String, details: String = "") {
                currentStage = stage
                currentStageStartedAt = logStageStart(stage, details)
            }

            fun completeStage(details: String = "") {
                logStageSuccess(currentStage, currentStageStartedAt, details)
            }

            fun rejectAtStage(code: String, message: String) {
                val e = IllegalStateException(message)
                logStageFailure(currentStage, currentStageStartedAt, "code=$code", e)
                promise.reject(code, message)
            }

            fun kvLayer0Shapes(kvCache: Map<String, OnnxTensor>): String {
                val key = kvCache["past_key_values.0.key"]?.let { tensorShape(it) } ?: "[missing]"
                val value = kvCache["past_key_values.0.value"]?.let { tensorShape(it) } ?: "[missing]"
                return "layer0_key=$key layer0_value=$value"
            }

            fun closeTrackedTensor(tensor: OnnxTensor?) {
                if (tensor == null) return
                try {
                    tensor.close()
                } finally {
                    tensorsToClose.remove(tensor)
                }
            }

            fun closeTrackedResult(result: OrtSession.Result?) {
                if (result == null) return
                try {
                    result.close()
                } finally {
                    resultsToClose.remove(result)
                }
            }

            try {
                if (isMemoryTight()) {
                    releaseObjectDetectionModelsInternal()
                    releaseAsrModelsInternal()
                }
                beginStage("VLM_STAGE_03 load_vision_session", "models=3")
                logAssetBytes("vision", "models/smolvlm2/smolvlm2_vision_hf.onnx")
                logAssetBytes("embed", "models/smolvlm2/smolvlm2_embed.onnx")
                logAssetBytes("decoder", "models/smolvlm2/smolvlm2_decoder.onnx")
                if (!autoLoadModel("smolvlm2_vis_hf", "models/smolvlm2/smolvlm2_vision_hf.onnx")) {
                    rejectAtStage("MODEL_NOT_LOADED", "SmolVLM2 model failed to load: models/smolvlm2/smolvlm2_vision_hf.onnx")
                    return@Thread
                }
                if (!autoLoadModel("smolvlm2_emb", "models/smolvlm2/smolvlm2_embed.onnx")) {
                    rejectAtStage("MODEL_NOT_LOADED", "SmolVLM2 model failed to load: models/smolvlm2/smolvlm2_embed.onnx")
                    return@Thread
                }
                if (!autoLoadModel("smolvlm2_dec", "models/smolvlm2/smolvlm2_decoder.onnx")) {
                    rejectAtStage("MODEL_NOT_LOADED", "SmolVLM2 model failed to load: models/smolvlm2/smolvlm2_decoder.onnx")
                    return@Thread
                }
                val visSession = sessions["smolvlm2_vis_hf"]!!
                val embSession = sessions["smolvlm2_emb"]!!
                val decSession = sessions["smolvlm2_dec"]!!
                completeStage(
                    "visionInputs=${visSession.inputNames.size} visionOutputs=${visSession.outputNames.size} " +
                        "embedInputs=${embSession.inputNames.size} embedOutputs=${embSession.outputNames.size} " +
                        "decoderInputs=${decSession.inputNames.size} decoderOutputs=${decSession.outputNames.size}"
                )

                beginStage("VLM_STAGE_01 decode_base64", "base64Chars=${base64Image.length}")
                val decoded = Base64.decode(base64Image, Base64.DEFAULT)
                bitmap = BitmapFactory.decodeByteArray(decoded, 0, decoded.size)
                    ?: run {
                        rejectAtStage("INVALID_IMAGE", "Cannot decode base64 to Bitmap")
                        return@Thread
                    }
                completeStage("decodedBytes=${decoded.size} bitmap=${bitmap!!.width}x${bitmap!!.height}")

                beginStage("VLM_STAGE_02 preprocess_image", "bitmap=${bitmap!!.width}x${bitmap!!.height}")
                val imageInputs = buildSmolVLM2ImageInputs(bitmap)
                val pvTensor = OnnxTensor.createTensor(
                    environment,
                    imageInputs.pixelValues,
                    longArrayOf(1, imageInputs.numImages.toLong(), 3, 512, 512)
                )
                val pamTensor = OnnxTensor.createTensor(environment, imageInputs.pixelAttentionMask)
                tensorsToClose += pvTensor; tensorsToClose += pamTensor
                completeStage(
                    "pixel_values shape=${tensorShape(pvTensor)} pixel_attention_mask shape=${tensorShape(pamTensor)} " +
                        "numImages=${imageInputs.numImages} rows=${imageInputs.rows} cols=${imageInputs.cols}"
                )

                beginStage("VLM_STAGE_05 tokenize_prompt", "questionChars=${question.length}")
                val tokenizer = SmolVLM2Tokenizer(reactContext)
                if (!tokenizer.isReady()) {
                    rejectAtStage("TOKENIZER_ERROR", "SmolVLM2 tokenizer failed")
                    return@Thread
                }
                val expandedPrompt = buildSmolVLM2ExpandedPrompt(question, imageInputs.rows, imageInputs.cols)
                val inputIds = tokenizer.encode(expandedPrompt)
                val imageTokenCount = inputIds.count { it == SmolVLM2Tokenizer.IMAGE_TOKEN_ID }
                val expectedImageTokenCount = imageInputs.numImages * 64
                if (imageTokenCount != expectedImageTokenCount) {
                    val message =
                        "Image token count mismatch: got $imageTokenCount, expected $expectedImageTokenCount"
                    rejectAtStage(
                        "VLM_PROMPT_ERROR",
                        message
                    )
                    return@Thread
                }
                completeStage(
                    "input_ids shape=[1,${inputIds.size}] tokenCount=${inputIds.size} " +
                        "imageTokens=$imageTokenCount promptChars=${expandedPrompt.length}"
                )

                android.util.Log.i(
                    tag,
                    "SmolVLM2 inputs: numImages=${imageInputs.numImages}, pixel_values=[1,${imageInputs.numImages},3,512,512], " +
                        "pixel_attention_mask=[1,${imageInputs.numImages},512,512], inputIds=${inputIds.size}, imageTokens=$imageTokenCount"
                )

                beginStage(
                    "VLM_STAGE_04 vision_inference",
                    "pixel_values shape=${tensorShape(pvTensor)} pixel_attention_mask shape=${tensorShape(pamTensor)}"
                )
                lastOnnxModel = "smolvlm2_vision_hf.onnx"
                lastOnnxInputs = "pixel_values=${tensorShape(pvTensor)}, pixel_attention_mask=${tensorShape(pamTensor)}"
                lastOnnxOutputs = visSession.outputNames.joinToString(",")
                val visRes = visSession.run(mapOf("pixel_values" to pvTensor, "pixel_attention_mask" to pamTensor))
                resultsToClose += visRes
                val imageFeats = visRes[0].value as Array<Array<FloatArray>>
                val imageFeatureBlocks = imageFeats.size
                val imageSeqLen = imageFeats[0].size
                val embedDim = imageFeats[0][0].size
                if (imageFeatureBlocks != imageInputs.numImages || imageSeqLen != 64) {
                    val message =
                        "Unexpected image_features shape: [$imageFeatureBlocks,$imageSeqLen,$embedDim]"
                    rejectAtStage(
                        "VLM_VISION_ERROR",
                        message
                    )
                    return@Thread
                }
                completeStage("image_features shape=${shape3D(imageFeats)} blocks=$imageFeatureBlocks seqLen=$imageSeqLen embedDim=$embedDim")
                closeTrackedTensor(pvTensor)
                closeTrackedTensor(pamTensor)

                beginStage("VLM_STAGE_06 embed_tokens", "input_ids shape=[1,${inputIds.size}] tokenCount=${inputIds.size}")
                val idsTensor = OnnxTensor.createTensor(
                    environment,
                    java.nio.LongBuffer.wrap(LongArray(inputIds.size) { inputIds[it].toLong() }),
                    longArrayOf(1, inputIds.size.toLong())
                )
                tensorsToClose += idsTensor
                lastOnnxModel = "smolvlm2_embed.onnx"
                lastOnnxInputs = "input_ids=${tensorShape(idsTensor)}"
                lastOnnxOutputs = embSession.outputNames.joinToString(",")
                val embRes = embSession.run(mapOf("input_ids" to idsTensor))
                resultsToClose += embRes
                val tokenEmbeds = embRes[0].value as Array<Array<FloatArray>>
                completeStage("input_ids shape=${tensorShape(idsTensor)} token_embeddings shape=${shape3D(tokenEmbeds)}")
                closeTrackedTensor(idsTensor)

                beginStage(
                    "VLM_STAGE_07 merge_image_embeddings",
                    "input_ids shape=[1,${inputIds.size}] image_features shape=${shape3D(imageFeats)} token_embeddings shape=${shape3D(tokenEmbeds)}"
                )
                val mergedFlat = mergeSmolVLM2Embeddings(inputIds, tokenEmbeds, imageFeats)
                closeTrackedResult(visRes)
                closeTrackedResult(embRes)

                val totalTokens = inputIds.size
                val embTensor = OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(mergedFlat),
                    longArrayOf(1, totalTokens.toLong(), embedDim.toLong())
                )
                tensorsToClose += embTensor
                completeStage("inputs_embeds shape=${tensorShape(embTensor)} mergedFloatCount=${mergedFlat.size}")
                android.util.Log.i(
                    tag,
                    "SmolVLM2 tensors: image_features=[$imageFeatureBlocks,$imageSeqLen,$embedDim], inputs_embeds=[1,$totalTokens,$embedDim]"
                )

                // Phase 4: Prefill — initialize empty KV-cache (30 layers x 2 = 60 tensors)
                beginStage("VLM_STAGE_08 decoder_prefill", "inputs_embeds shape=${tensorShape(embTensor)}")
                val numLayers = 30
                val kvCache   = mutableMapOf<String, OnnxTensor>()
                for (layer in 0 until numLayers) for (kv in listOf("key", "value")) {
                    val t = OnnxTensor.createTensor(environment, FloatBuffer.allocate(0), longArrayOf(1, 3, 0, 64))
                    kvCache["past_key_values.$layer.$kv"] = t; tensorsToClose += t
                }
                val posArr    = LongArray(totalTokens) { it.toLong() }
                val maskArr   = LongArray(totalTokens) { 1L }
                val posTensor = OnnxTensor.createTensor(environment, java.nio.LongBuffer.wrap(posArr),  longArrayOf(1, totalTokens.toLong()))
                val mskTensor = OnnxTensor.createTensor(environment, java.nio.LongBuffer.wrap(maskArr), longArrayOf(1, totalTokens.toLong()))
                tensorsToClose += posTensor; tensorsToClose += mskTensor
                val prefillFeed = mutableMapOf<String, OnnxTensor>(
                    "inputs_embeds"  to embTensor,
                    "attention_mask" to mskTensor,
                    "position_ids"   to posTensor
                ).also { it.putAll(kvCache) }
                logRuntimeMemory("before decoder execution smolvlm2_decoder.onnx prefill")
                android.util.Log.i(
                    tag,
                    "VLM_STAGE_08 decoder_prefill status=before_onnx elapsedMs=0 " +
                        "inputs_embeds shape=${tensorShape(embTensor)} attention_mask shape=${tensorShape(mskTensor)} " +
                        "position_ids shape=${tensorShape(posTensor)} KV-cache count=${kvCache.size} ${kvLayer0Shapes(kvCache)}"
                )
                lastOnnxModel = "smolvlm2_decoder.onnx"
                lastOnnxInputs = "inputs_embeds=${tensorShape(embTensor)}, attention_mask=${tensorShape(mskTensor)}, position_ids=${tensorShape(posTensor)}, kvCache=${kvCache.size}"
                lastOnnxOutputs = decSession.outputNames.joinToString(",")
                val prefillRes  = decSession.run(prefillFeed); resultsToClose += prefillRes
                val outNames    = decSession.outputNames.toList()
                val prefLogits  = prefillRes[0].value as Array<Array<FloatArray>>
                var nextLogits  = prefLogits[0][prefLogits[0].size - 1]  // [49280]

                fun refreshKv(res: OrtSession.Result): Map<String, OnnxTensor> {
                    val m = mutableMapOf<String, OnnxTensor>()
                    for (i in 1 until outNames.size) {
                        val past = outNames[i].replace("present.", "past_key_values.")
                        val arr  = res[i].value as Array<Array<Array<FloatArray>>>
                        val t    = OnnxTensor.createTensor(environment, arr)
                        m[past]  = t; tensorsToClose += t
                    }
                    return m
                }
                kvCache.values.forEach { closeTrackedTensor(it) }; kvCache.clear(); kvCache.putAll(refreshKv(prefillRes))
                completeStage(
                    "logits shape=${shape3D(prefLogits)} nextLogits count=${nextLogits.size} " +
                        "KV-cache count=${kvCache.size} ${kvLayer0Shapes(kvCache)}"
                )
                closeTrackedResult(prefillRes)
                closeTrackedTensor(embTensor)
                closeTrackedTensor(posTensor)
                closeTrackedTensor(mskTensor)

                // Phase 5: Autoregressive generation with KV-cache
                beginStage("VLM_STAGE_09 decoder_generation", "maxTokens=64 embedDim=$embedDim initialPastLen=$totalTokens")
                val EOS_ID = 49279; val MAX_TOKENS = 64
                val generatedIds = mutableListOf<Int>()
                var pastLen = totalTokens
                for (step in 0 until MAX_TOKENS) {
                    val nextTok = nextLogits.indices.maxByOrNull { nextLogits[it] } ?: EOS_ID
                    if (nextTok == EOS_ID) break
                    generatedIds.add(nextTok)
                    val tokBuf = java.nio.LongBuffer.wrap(longArrayOf(nextTok.toLong()))
                    val tokIdT = OnnxTensor.createTensor(environment, tokBuf, longArrayOf(1, 1)); tensorsToClose += tokIdT
                    android.util.Log.i(
                        tag,
                        "VLM_STAGE_09 decoder_generation status=before_embed elapsedMs=0 step=$step input_ids shape=${tensorShape(tokIdT)}"
                    )
                    lastOnnxModel = "smolvlm2_embed.onnx"
                    lastOnnxInputs = "input_ids=${tensorShape(tokIdT)} step=$step"
                    lastOnnxOutputs = embSession.outputNames.joinToString(",")
                    val tokEmbR = embSession.run(mapOf("input_ids" to tokIdT)); resultsToClose += tokEmbR
                    val tokEmb  = tokEmbR[0].value as Array<Array<FloatArray>>
                    val teBuf   = FloatBuffer.allocate(embedDim)
                    for (d in 0 until embedDim) teBuf.put(tokEmb[0][0][d])
                    teBuf.rewind()
                    closeTrackedResult(tokEmbR)
                    closeTrackedTensor(tokIdT)
                    val teT = OnnxTensor.createTensor(environment, teBuf, longArrayOf(1, 1, embedDim.toLong())); tensorsToClose += teT
                    val npBuf = java.nio.LongBuffer.wrap(longArrayOf(pastLen.toLong()))
                    val nmBuf = java.nio.LongBuffer.wrap(LongArray(pastLen + 1) { 1L })
                    val npT   = OnnxTensor.createTensor(environment, npBuf, longArrayOf(1, 1)); tensorsToClose += npT
                    val nmT   = OnnxTensor.createTensor(environment, nmBuf, longArrayOf(1, (pastLen + 1).toLong())); tensorsToClose += nmT
                    val decFeed = mutableMapOf<String, OnnxTensor>(
                        "inputs_embeds" to teT, "attention_mask" to nmT, "position_ids" to npT
                    ).also { it.putAll(kvCache) }
                    logRuntimeMemory("before decoder execution smolvlm2_decoder.onnx generation step=$step")
                    android.util.Log.i(
                        tag,
                        "VLM_STAGE_09 decoder_generation status=before_decoder elapsedMs=0 step=$step " +
                            "inputs_embeds shape=${tensorShape(teT)} attention_mask shape=${tensorShape(nmT)} " +
                            "position_ids shape=${tensorShape(npT)} KV-cache count=${kvCache.size} ${kvLayer0Shapes(kvCache)}"
                    )
                    lastOnnxModel = "smolvlm2_decoder.onnx"
                    lastOnnxInputs = "inputs_embeds=${tensorShape(teT)}, attention_mask=${tensorShape(nmT)}, position_ids=${tensorShape(npT)}, kvCache=${kvCache.size}, step=$step"
                    lastOnnxOutputs = decSession.outputNames.joinToString(",")
                    val decRes = decSession.run(decFeed); resultsToClose += decRes
                    nextLogits = (decRes[0].value as Array<Array<FloatArray>>)[0][0]
                    kvCache.values.forEach { closeTrackedTensor(it) }; kvCache.clear(); kvCache.putAll(refreshKv(decRes))
                    android.util.Log.i(
                        tag,
                        "VLM_STAGE_09 decoder_generation status=step_success elapsedMs=0 step=$step " +
                            "logits shape=[1,1,${nextLogits.size}] generatedCount=${generatedIds.size} " +
                            "KV-cache count=${kvCache.size} ${kvLayer0Shapes(kvCache)}"
                    )
                    closeTrackedResult(decRes)
                    closeTrackedTensor(teT)
                    closeTrackedTensor(npT)
                    closeTrackedTensor(nmT)
                    pastLen++
                }
                completeStage("generatedCount=${generatedIds.size} finalPastLen=$pastLen")
                android.util.Log.i(
                    tag,
                    "SmolVLM2 decode: logits=[1,1,49280], generatedCount=${generatedIds.size}"
                )

                beginStage("VLM_STAGE_10 decode_text", "generatedCount=${generatedIds.size}")
                val answer = tokenizer.decode(generatedIds.toIntArray())
                completeStage("answerChars=${answer.length} generatedCount=${generatedIds.size}")
                promise.resolve(Arguments.createMap().apply {
                    putString("answer", answer)
                    putBoolean("isSuccess", true)
                    putString("model", "SmolVLM2-256M-HF-INT8")
                    putInt("tokenCount", generatedIds.size)
                })

            } catch (e: Exception) {
                val ortDetails = if (e is OrtException) {
                    val code = try {
                        e.javaClass.getMethod("getCode").invoke(e)?.toString()
                    } catch (_: Exception) {
                        "unavailable"
                    }
                    "ortMessage=${e.message} ortCode=$code failingModel=$lastOnnxModel failingInputs=$lastOnnxInputs failingOutputs=$lastOnnxOutputs"
                } else {
                    "failingModel=$lastOnnxModel failingInputs=$lastOnnxInputs failingOutputs=$lastOnnxOutputs"
                }
                logStageFailure(currentStage, currentStageStartedAt, ortDetails, e)
                android.util.Log.e(tag, "VLM_RUNTIME_ERROR stage=$currentStage $ortDetails", e)
                promise.reject("VLM_ERROR", "SmolVLM2 inference failed: ${e.message}", e)
            } finally {
                try { resultsToClose.forEach { it.close() }; tensorsToClose.forEach { it.close() }
                      bitmap?.recycle()
                } catch (_: Exception) {}
            }
        }.start()
    }


    @ReactMethod
    fun runSpeechRecognition(base64Audio: String, promise: Promise) {
        // Verified ONNX I/O from graph inspection:
        //   phowhisper_encoder.onnx: input='mel_input' [batch,80,mel_time] FLOAT
        //                            output='encoder_hidden_states' [batch,seq_len,384] FLOAT
        //   phowhisper_decoder.onnx: input='token_ids' [batch,target_len] INT64
        //                            input='encoder_hidden_states' [batch,source_len,384] FLOAT
        //                            output='logits' [batch,target_len,51865] FLOAT (no KV-cache)
        // vocab_size=51865 matches phowhisper vocabulary.json (51,865 tokens)
        Thread {
            var melTensor: OnnxTensor? = null
            var encOutput: ai.onnxruntime.OrtSession.Result? = null
            var decOutput: ai.onnxruntime.OrtSession.Result? = null
            var decInputIdsTensor: OnnxTensor? = null
            var decEncStateTensor: OnnxTensor? = null
            try {
                if (isMemoryTight()) {
                    releaseObjectDetectionModelsInternal()
                    releaseVlmModelsInternal()
                }
                autoLoadModel("phowhisper_encoder", "models/phowhisper/phowhisper_encoder.onnx")
                autoLoadModel("phowhisper_decoder", "models/phowhisper/phowhisper_decoder.onnx")
                val encSession = sessions["phowhisper_encoder"]
                val decSession = sessions["phowhisper_decoder"]
                if (encSession == null) {
                    promise.reject("MODEL_NOT_LOADED", "phowhisper_encoder failed to load")
                    return@Thread
                }
                if (decSession == null) {
                    promise.reject("MODEL_NOT_LOADED", "phowhisper_decoder failed to load")
                    return@Thread
                }

                // --- Phase 1: Audio preprocessing ---
                val decodedAudio = Base64.decode(base64Audio, Base64.DEFAULT)
                val pcmData = ShortArray(decodedAudio.size / 2)
                java.nio.ByteBuffer.wrap(decodedAudio)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asShortBuffer().get(pcmData)

                val melData = AudioPreProcessor.computeLogMelSpectrogram(pcmData)
                val melBuffer = FloatBuffer.allocate(1 * 80 * 3000)
                melBuffer.put(melData)
                melBuffer.rewind()

                // --- Phase 2: Encoder inference ---
                // Input: 'mel_input' [1, 80, 3000] FLOAT
                melTensor = OnnxTensor.createTensor(environment, melBuffer, longArrayOf(1, 80, 3000))
                encOutput = encSession.run(mapOf("mel_input" to melTensor))
                // Output: 'encoder_hidden_states' [1, seq_len, 384] FLOAT
                val encHidden = encOutput[0].value as Array<Array<FloatArray>>
                val sourceLen = encHidden[0].size
                val hiddenDim = encHidden[0][0].size // 384

                // --- Phase 3: Decoder greedy decode loop ---
                // Inputs:
                //   'token_ids' [1, target_len] INT64 — grows each step
                //   'encoder_hidden_states' [1, source_len, 384] FLOAT — constant
                // Output: 'logits' [1, target_len, 51865] FLOAT
                val detokenizer = PhoWhisperDetokenizer(reactContext)
                if (!detokenizer.isReady()) {
                    promise.reject("TOKENIZER_ERROR", "PhoWhisper vocab not loaded")
                    return@Thread
                }

                // Flatten encoder_hidden_states for reuse across decode steps
                val encHiddenFlat = FloatBuffer.allocate(sourceLen * hiddenDim)
                for (s in 0 until sourceLen) {
                    for (d in 0 until hiddenDim) {
                        encHiddenFlat.put(encHidden[0][s][d])
                    }
                }

                // Whisper decode prompt: <startoftranscript><vi><transcribe><notimestamps>
                val EOT_TOKEN = 50256  // <|endoftext|>
                val MAX_TOKENS = 100
                val generatedIds = mutableListOf<Int>()
                // Current decoder input sequence (grows each step)
                var currentTokenIds = intArrayOf(50258, 50284, 50359, 50363)

                for (step in 0 until MAX_TOKENS) {
                    val targetLen = currentTokenIds.size.toLong()
                    val inputIdsLong = LongArray(currentTokenIds.size) { currentTokenIds[it].toLong() }

                    decInputIdsTensor = OnnxTensor.createTensor(
                        environment,
                        java.nio.LongBuffer.wrap(inputIdsLong),
                        longArrayOf(1, targetLen)
                    )
                    encHiddenFlat.rewind()
                    decEncStateTensor = OnnxTensor.createTensor(
                        environment, encHiddenFlat,
                        longArrayOf(1, sourceLen.toLong(), hiddenDim.toLong())
                    )

                    decOutput = decSession.run(mapOf(
                        "token_ids"              to decInputIdsTensor!!,
                        "encoder_hidden_states"  to decEncStateTensor!!
                    ))

                    // logits [1, target_len, 51865] — take last position
                    val logits = decOutput[0].value as Array<Array<FloatArray>>
                    val lastPos = logits[0].size - 1
                    val vocabLogits = logits[0][lastPos]

                    var maxLogit = Float.NEGATIVE_INFINITY
                    var nextToken = EOT_TOKEN
                    for (i in vocabLogits.indices) {
                        if (vocabLogits[i] > maxLogit) {
                            maxLogit = vocabLogits[i]
                            nextToken = i
                        }
                    }

                    decOutput.close(); decOutput = null
                    decInputIdsTensor.close(); decInputIdsTensor = null
                    decEncStateTensor.close(); decEncStateTensor = null

                    if (nextToken == EOT_TOKEN) break
                    generatedIds.add(nextToken)
                    currentTokenIds = currentTokenIds + nextToken
                }

                val transcript = detokenizer.decode(generatedIds.toIntArray())

                val resultMap = Arguments.createMap().apply {
                    putString("text", transcript)
                    // matchedState is determined upstream by the JS voice command matcher, not hardcoded here
                }
                promise.resolve(resultMap)

            } catch (e: Exception) {
                promise.reject("ASR_ERROR", "PhoWhisper inference failed: ${e.message}", e)
            } finally {
                try {
                    decOutput?.close()
                    decInputIdsTensor?.close()
                    decEncStateTensor?.close()
                    encOutput?.close()
                    melTensor?.close()
                } catch (_: Exception) {}
            }
        }.start()
    }

    @ReactMethod
    fun closeSession(modelId: String, promise: Promise) {
        try {
            val session = sessions.remove(modelId)
            session?.close()
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("CLOSE_ERROR", e.message, e)
        }
    }
}
