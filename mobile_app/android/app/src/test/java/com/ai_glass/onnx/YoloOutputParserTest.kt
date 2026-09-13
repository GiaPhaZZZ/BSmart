package com.ai_glass.onnx

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloOutputParserTest {

    @Test
    fun parsesRealYoloXyxyConfidenceClassOutput() {
        val rows = Array(300) { FloatArray(6) }
        rows[0] = floatArrayOf(100f, 50f, 300f, 450f, 0.91f, 56f)
        rows[1] = floatArrayOf(110f, 60f, 310f, 460f, 0.80f, 56f)
        rows[2] = floatArrayOf(110f, 60f, 310f, 460f, 0.72f, 60f)
        rows[3] = floatArrayOf(20f, 20f, 40f, 40f, 0.49f, 0f)

        val parsed = YoloOutputParser.parse(arrayOf(rows), confidenceThreshold = 0.5f)

        assertEquals(2, parsed.size)

        val chair = parsed[0]
        assertEquals(56, chair.classId)
        assertEquals(0.91f, chair.score, 0.0001f)
        assertEquals(200f / 640f, chair.cx, 0.0001f)
        assertEquals(250f / 640f, chair.cy, 0.0001f)
        assertEquals(200f / 640f, chair.w, 0.0001f)
        assertEquals(400f / 640f, chair.h, 0.0001f)

        val table = parsed[1]
        assertEquals(60, table.classId)
        assertEquals(0.72f, table.score, 0.0001f)
    }

    @Test
    fun keepsLegacyAnchorClassScoreOutputSupported() {
        val output = Array(84) { FloatArray(2) }
        output[0][0] = 320f
        output[1][0] = 160f
        output[2][0] = 128f
        output[3][0] = 64f
        output[4 + 0][0] = 0.93f

        output[0][1] = 0.25f
        output[1][1] = 0.75f
        output[2][1] = 0.10f
        output[3][1] = 0.20f
        output[4 + 2][1] = 0.88f

        val parsed = YoloOutputParser.parse(arrayOf(output), confidenceThreshold = 0.5f)

        assertEquals(2, parsed.size)
        assertEquals(0, parsed[0].classId)
        assertEquals(0.93f, parsed[0].score, 0.0001f)
        assertEquals(0.5f, parsed[0].cx, 0.0001f)
        assertEquals(0.25f, parsed[0].cy, 0.0001f)
        assertEquals(0.2f, parsed[0].w, 0.0001f)
        assertEquals(0.1f, parsed[0].h, 0.0001f)

        assertEquals(2, parsed[1].classId)
        assertEquals(0.88f, parsed[1].score, 0.0001f)
        assertEquals(0.25f, parsed[1].cx, 0.0001f)
        assertEquals(0.75f, parsed[1].cy, 0.0001f)
    }

    @Test
    fun rejectsUnsupportedOutputShape() {
        val output = Array(10) { FloatArray(10) }

        try {
            YoloOutputParser.parse(arrayOf(output))
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message.orEmpty().contains("Unsupported YOLO output shape"))
            return
        }

        throw AssertionError("Expected unsupported YOLO shape to throw")
    }
}
