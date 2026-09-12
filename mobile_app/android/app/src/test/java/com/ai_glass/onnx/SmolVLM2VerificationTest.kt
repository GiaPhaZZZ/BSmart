package com.ai_glass.onnx

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import com.facebook.react.bridge.BridgeReactContext
import com.facebook.react.bridge.ReactApplicationContext
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class SmolVLM2VerificationTest {

    private val appContext: Context
        get() = RuntimeEnvironment.getApplication()

    private val reactContext: ReactApplicationContext
        get() = BridgeReactContext(appContext)

    private fun tokenizer(): SmolVLM2Tokenizer = SmolVLM2Tokenizer(appContext)

    private fun module(): OnnxInferenceModule = OnnxInferenceModule(reactContext)

    @Test
    fun tokenizerMatchesHuggingFaceReferenceIds() {
        val tokenizer = tokenizer()
        assertTrue(tokenizer.isReady())

        val cases = listOf(
            "<|im_start|>" to intArrayOf(1),
            "User:" to intArrayOf(11126, 42),
            "<image>" to intArrayOf(49190),
            "<end_of_utterance>" to intArrayOf(49279),
            "Assistant:" to intArrayOf(9519, 9531, 42),
            "<|im_start|>\nUser:\n<image>\n<end_of_utterance>\nAssistant:" to
                intArrayOf(1, 198, 11126, 42, 198, 49190, 198, 49279, 198, 9519, 9531, 42),
            "<|im_start|>User:<image><end_of_utterance>\nUser: Is there an obstacle ahead?<end_of_utterance>\nAssistant:" to
                intArrayOf(1, 11126, 42, 49190, 49279, 198, 11126, 42, 1431, 665, 354, 20135, 5976, 47, 49279, 198, 9519, 9531, 42)
        )

        for ((text, expectedIds) in cases) {
            assertArrayEquals("Token IDs mismatch for $text", expectedIds, tokenizer.encode(text))
        }
    }

    @Test
    fun expandedPromptFor512ImageMatchesHuggingFaceCounts() {
        val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        try {
            val module = module()
            val imageInputs = module.buildSmolVLM2ImageInputs(bitmap)
            val inputIds = tokenizer().encode(
                module.buildSmolVLM2ExpandedPrompt(
                    "Is there an obstacle ahead?",
                    imageInputs.rows,
                    imageInputs.cols
                )
            )

            assertEquals(17, imageInputs.numImages)
            assertEquals(1088, inputIds.count { it == SmolVLM2Tokenizer.IMAGE_TOKEN_ID })
            assertEquals(18, inputIds.count { it == SmolVLM2Tokenizer.FAKE_IMAGE_TOKEN_ID })
            assertEquals(1, inputIds.count { it == SmolVLM2Tokenizer.GLOBAL_IMAGE_TOKEN_ID })
            assertEquals(16, inputIds.count { it in 49153..49188 })
            assertEquals(1145, inputIds.size)
        } finally {
            bitmap.recycle()
        }
    }

    @Test
    fun embeddingMergeReplacesOnlyImageTokenPositionsInOrder() {
        val inputIds = intArrayOf(10, SmolVLM2Tokenizer.IMAGE_TOKEN_ID, 11, SmolVLM2Tokenizer.IMAGE_TOKEN_ID, 12)
        val tokenEmbeds = Array(1) { Array(inputIds.size) { tokenIndex -> FloatArray(3) { dim -> (tokenIndex * 10 + dim).toFloat() } } }
        val imageFeats = arrayOf(
            arrayOf(
                floatArrayOf(1000f, 1001f, 1002f),
                floatArrayOf(2000f, 2001f, 2002f)
            )
        )

        val merged = module().mergeSmolVLM2Embeddings(inputIds, tokenEmbeds, imageFeats)

        assertArrayEquals(floatArrayOf(0f, 1f, 2f), merged.sliceArray(0..2), 0.0f)
        assertArrayEquals(floatArrayOf(1000f, 1001f, 1002f), merged.sliceArray(3..5), 0.0f)
        assertArrayEquals(floatArrayOf(20f, 21f, 22f), merged.sliceArray(6..8), 0.0f)
        assertArrayEquals(floatArrayOf(2000f, 2001f, 2002f), merged.sliceArray(9..11), 0.0f)
        assertArrayEquals(floatArrayOf(40f, 41f, 42f), merged.sliceArray(12..14), 0.0f)
    }

    @Test
    fun preDecoderTensorShapesFor512ImageMatchAcceptanceContract() {
        val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        try {
            val module = module()
            val imageInputs = module.buildSmolVLM2ImageInputs(bitmap)
            val inputIds = tokenizer().encode(
                module.buildSmolVLM2ExpandedPrompt(
                    "Is there an obstacle ahead?",
                    imageInputs.rows,
                    imageInputs.cols
                )
            )

            val imageFeatures = Array(17) { imageIndex ->
                Array(64) { localIndex ->
                    FloatArray(576) { dim -> imageIndex * 100000f + localIndex * 1000f + dim }
                }
            }
            val tokenEmbeds = Array(1) {
                Array(inputIds.size) { tokenIndex ->
                    FloatArray(576) { dim -> tokenIndex * 1000f + dim }
                }
            }
            val inputsEmbeds = module.mergeSmolVLM2Embeddings(inputIds, tokenEmbeds, imageFeatures)
            val attentionMask = LongArray(inputIds.size) { 1L }
            val positionIds = LongArray(inputIds.size) { it.toLong() }

            assertEquals(listOf(1, 17, 3, 512, 512), listOf(1, imageInputs.numImages, 3, 512, 512))
            assertEquals(17 * 3 * 512 * 512, imageInputs.pixelValues.capacity())
            assertEquals(listOf(1, 17, 512, 512), listOf(1, imageInputs.pixelAttentionMask[0].size, imageInputs.pixelAttentionMask[0][0].size, imageInputs.pixelAttentionMask[0][0][0].size))
            assertEquals(listOf(17, 64, 576), listOf(imageFeatures.size, imageFeatures[0].size, imageFeatures[0][0].size))
            assertEquals(listOf(1, 1145, 576), listOf(1, inputIds.size, 576))
            assertEquals(1145 * 576, inputsEmbeds.size)
            assertEquals(listOf(1, 1145), listOf(1, attentionMask.size))
            assertEquals(listOf(1, 1145), listOf(1, positionIds.size))
        } finally {
            bitmap.recycle()
        }
    }
}
