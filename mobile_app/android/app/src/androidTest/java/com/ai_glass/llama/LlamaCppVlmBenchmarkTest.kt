package com.ai_glass.llama

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.ceil

@RunWith(AndroidJUnit4::class)
class LlamaCppVlmBenchmarkTest {
    private val tag = "BSmartGgufBench"
    private val maxTokens = 16

    @Test
    fun runSingleVisualQaRequest() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getArguments()
        val ggufDir = File(context.filesDir, "gguf")
        val model = File(
            ggufDir,
            args.getString("modelFile") ?: "SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        )
        val mmproj = File(
            ggufDir,
            args.getString("mmprojFile") ?: "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        )
        val contextSize = args.getString("contextSize")?.toIntOrNull() ?: 2048
        val threads = args.getString("threads")?.toIntOrNull() ?: 4
        val singleMaxTokens = args.getString("maxTokens")?.toIntOrNull() ?: 16
        val imageSize = args.getString("imageSize")?.toIntOrNull() ?: 128
        val imageMaxTokens = args.getString("imageMaxTokens")?.toIntOrNull() ?: 512
        LlamaCppVlmNative.setImageMaxTokensNative(imageMaxTokens)

        assertTrue("GGUF missing: ${model.absolutePath}", model.isFile)
        assertTrue("mmproj missing: ${mmproj.absolutePath}", mmproj.isFile)

        val imageFileName = args.getString("imageFile")
        val image = if (imageFileName != null) {
            File(context.cacheDir, imageFileName)
        } else {
            val img = File(context.cacheDir, "gguf_single_image_${imageSize}.jpg")
            writeBenchmarkImage(img, imageSize)
            img
        }
        val prompt = args.getString("prompt") ?: "What is in front of me? Answer in a few words."
        Log.i(
            tag,
            "SINGLE_CONFIG model=${model.name} mmproj=${mmproj.name} image=${image.name} " +
                "contextSize=$contextSize threads=$threads maxTokens=$singleMaxTokens imageMaxTokens=$imageMaxTokens"
        )

        val loadStart = System.nanoTime()
        val loadJson = JSONObject(
            LlamaCppVlmNative.loadModelNative(
                model.absolutePath,
                mmproj.absolutePath,
                contextSize,
                threads,
                singleMaxTokens,
                0.0
            )
        )
        val loadMs = elapsedMs(loadStart)
        Log.i(tag, "SINGLE_COLD_LOAD model_load_ms=$loadMs json=$loadJson")
        assertTrue(loadJson.toString(), loadJson.optBoolean("isSuccess", false))

        Log.i(tag, "SINGLE_RUN_START image=${image.name} prompt=$prompt")
        val runJson = JSONObject(
            LlamaCppVlmNative.runVisualQANative(
                image.absolutePath,
                prompt,
                singleMaxTokens,
                0.0
            )
        )
        Log.i(tag, "SINGLE_RUN_RESULT json=$runJson")
        assertTrue(runJson.toString(), runJson.optBoolean("isSuccess", false))
        assertFalse("Single answer is blank", runJson.optString("answer").trim().isEmpty())

        val unloadJson = JSONObject(LlamaCppVlmNative.unloadModelNative())
        assertEquals(unloadJson.toString(), true, unloadJson.optBoolean("isSuccess", false))
    }

    @Test
    fun runTenWarmVisualQaRequests() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getArguments()
        val ggufDir = File(context.filesDir, "gguf")
        val model = File(
            ggufDir,
            args.getString("modelFile") ?: "SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf"
        )
        val mmproj = File(
            ggufDir,
            args.getString("mmprojFile") ?: "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        )
        val contextSize = args.getString("contextSize")?.toIntOrNull() ?: 2048
        val threads = args.getString("threads")?.toIntOrNull() ?: 6
        val singleMaxTokens = args.getString("maxTokens")?.toIntOrNull() ?: 16
        val imageSize = args.getString("imageSize")?.toIntOrNull() ?: 128
        val imageMaxTokens = args.getString("imageMaxTokens")?.toIntOrNull() ?: 128
        val numRuns = args.getString("numRuns")?.toIntOrNull() ?: 5
        LlamaCppVlmNative.setImageMaxTokensNative(imageMaxTokens)

        assertTrue("GGUF missing: ${model.absolutePath}", model.isFile)
        assertTrue("mmproj missing: ${mmproj.absolutePath}", mmproj.isFile)

        val image = File(context.cacheDir, "gguf_single_image_${imageSize}.jpg")
        writeBenchmarkImage(image, imageSize)

        val loadStart = System.nanoTime()
        val loadJson = JSONObject(
            LlamaCppVlmNative.loadModelNative(
                model.absolutePath,
                mmproj.absolutePath,
                contextSize,
                threads,
                singleMaxTokens,
                0.0
            )
        )
        val loadMs = elapsedMs(loadStart)
        assertTrue(loadJson.toString(), loadJson.optBoolean("isSuccess", false))
        Log.i(tag, "WARM_COLD_LOAD model_load_ms=$loadMs json=$loadJson")

        val prompt = "What is in front of me? Answer in a few words."
        Log.i(tag, "WARMUP_START")
        val warmup = JSONObject(LlamaCppVlmNative.runVisualQANative(image.absolutePath, prompt, singleMaxTokens, 0.0))
        assertTrue(warmup.toString(), warmup.optBoolean("isSuccess", false))
        assertFalse("Warmup answer is blank", warmup.optString("answer").trim().isEmpty())
        Log.i(tag, "WARMUP json=$warmup")

        val totals = mutableListOf<Long>()
        val visions = mutableListOf<Long>()
        val generations = mutableListOf<Long>()
        val prompts = mutableListOf<Long>()
        val ttfts = mutableListOf<Long>()
        val tokenRates = mutableListOf<Double>()
        var totalTokens = 0
        for (run in 1..numRuns) {
            Log.i(tag, "RUN $run START")
            val runJson = JSONObject(LlamaCppVlmNative.runVisualQANative(image.absolutePath, prompt, singleMaxTokens, 0.0))
            assertTrue(runJson.toString(), runJson.optBoolean("isSuccess", false))
            val answer = runJson.optString("answer").trim()
            assertFalse("Run $run answer is blank", answer.isEmpty())

            val timings = runJson.getJSONObject("timings")
            val imagePrep = timings.optLong("image_preprocessing_ms")
            val vision = timings.optLong("vision_encoding_ms")
            val promptMs = timings.optLong("prompt_processing_ms")
            val generation = timings.optLong("generation_ms")
            val total = timings.optLong("total_vlm_ms")
            val tokens = runJson.optInt("generatedTokens")
            val tps = runJson.optDouble("tokensPerSecond")
            val ttft = runJson.optLong("timeToFirstTokenMs")

            totals += total
            visions += vision
            generations += generation
            prompts += promptMs
            ttfts += ttft
            tokenRates += tps
            totalTokens += tokens
            Log.i(
                tag,
                "RUN $run image_prep_ms=$imagePrep vision_ms=$vision prompt_ms=$promptMs " +
                    "generation_ms=$generation total_ms=$total ttft_ms=$ttft tokens=$tokens " +
                    "tok_s=$tps answer=$answer"
            )
        }

        Log.i(
            tag,
            "WARM_BENCHMARK_SUMMARY runs=$numRuns " +
                "vision_min=${visions.min()} vision_median=${median(visions)} vision_avg=${visions.average()} vision_max=${visions.max()} " +
                "gen_min=${generations.min()} gen_median=${median(generations)} gen_avg=${generations.average()} gen_max=${generations.max()} " +
                "total_min=${totals.min()} total_median=${median(totals)} total_avg=${totals.average()} total_max=${totals.max()} " +
                "avg_tok_s=${tokenRates.average()} total_tokens=$totalTokens"
        )

        val unloadJson = JSONObject(LlamaCppVlmNative.unloadModelNative())
        assertEquals(unloadJson.toString(), true, unloadJson.optBoolean("isSuccess", false))
    }

    private fun writeBenchmarkImage(file: File, size: Int) {
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)

        val scale = size / 512f
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.RED
        canvas.drawRect(48f * scale, 80f * scale, 250f * scale, 300f * scale, paint)
        paint.color = Color.BLUE
        canvas.drawCircle(365f * scale, 190f * scale, 95f * scale, paint)
        paint.color = Color.BLACK
        paint.textSize = (38f * scale).coerceAtLeast(10f)
        canvas.drawText("RED BOX", 70f * scale, 370f * scale, paint)
        canvas.drawText("BLUE CIRCLE", 235f * scale, 430f * scale, paint)

        file.outputStream().use { stream ->
            assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream))
        }
        bitmap.recycle()
    }

    private fun elapsedMs(startNs: Long): Long =
        (System.nanoTime() - startNs) / 1_000_000L

    private fun median(values: List<Long>): Double {
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid].toDouble()
        }
    }

    private fun p90(values: List<Long>): Long {
        val sorted = values.sorted()
        return sorted[(ceil(sorted.size * 0.9).toInt() - 1).coerceIn(sorted.indices)]
    }
}
