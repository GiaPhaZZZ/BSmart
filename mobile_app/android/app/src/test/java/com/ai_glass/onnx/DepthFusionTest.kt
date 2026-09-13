package com.ai_glass.onnx

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DepthFusionTest {

    @Test
    fun returnsDifferentDepthsForBoxesInDifferentDepthRegions() {
        val depthMap = Array(10) { row ->
            FloatArray(10) { col -> if (col < 5) 0.21f else 0.63f + row * 0.001f }
        }
        val leftBox = YoloDetectionBox(classId = 0, score = 0.9f, cx = 0.25f, cy = 0.5f, w = 0.2f, h = 0.4f)
        val rightBox = YoloDetectionBox(classId = 56, score = 0.9f, cx = 0.75f, cy = 0.5f, w = 0.2f, h = 0.4f)

        val leftDepth = DepthFusion.relativeDepthForBox(leftBox, depthMap)
        val rightDepth = DepthFusion.relativeDepthForBox(rightBox, depthMap)

        assertNotNull(leftDepth)
        assertNotNull(rightDepth)
        assertEquals(0.21f, leftDepth!!, 0.0001f)
        assertTrue(rightDepth!! > leftDepth)
        assertNotEquals(leftDepth, rightDepth)
    }

    @Test
    fun clampsOutOfBoundsBoxToDepthMapBounds() {
        val depthMap = Array(10) { FloatArray(10) { 0.47f } }
        val box = YoloDetectionBox(classId = 0, score = 0.9f, cx = 0.05f, cy = 0.5f, w = 0.5f, h = 0.5f)

        val region = DepthFusion.bboxToDepthRegion(box, depthMap)
        val depth = DepthFusion.relativeDepthForBox(box, depthMap)

        assertNotNull(region)
        assertEquals(0, region!!.left)
        assertTrue(region.rightExclusive <= 10)
        assertTrue(region.top >= 0)
        assertTrue(region.bottomExclusive <= 10)
        assertEquals(0.47f, depth!!, 0.0001f)
    }

    @Test
    fun returnsNullForInvalidOrTooSmallBoxes() {
        val depthMap = Array(10) { FloatArray(10) { 0.5f } }
        val zeroWidth = YoloDetectionBox(classId = 0, score = 0.9f, cx = 0.5f, cy = 0.5f, w = 0f, h = 0.4f)
        val offscreen = YoloDetectionBox(classId = 0, score = 0.9f, cx = 1.2f, cy = 0.5f, w = 0.1f, h = 0.4f)
        val tiny = YoloDetectionBox(classId = 0, score = 0.9f, cx = 0.51f, cy = 0.51f, w = 0.001f, h = 0.001f)

        assertNull(DepthFusion.relativeDepthForBox(zeroWidth, depthMap))
        assertNull(DepthFusion.relativeDepthForBox(offscreen, depthMap))
        assertNull(DepthFusion.relativeDepthForBox(tiny, depthMap))
    }

    @Test
    fun returnsSameObjectDepthsForConstantDepthMap() {
        val depthMap = Array(10) { FloatArray(10) { 0.35f } }
        val firstBox = YoloDetectionBox(classId = 0, score = 0.9f, cx = 0.25f, cy = 0.35f, w = 0.3f, h = 0.3f)
        val secondBox = YoloDetectionBox(classId = 60, score = 0.8f, cx = 0.75f, cy = 0.65f, w = 0.3f, h = 0.3f)

        val firstDepth = DepthFusion.relativeDepthForBox(firstBox, depthMap)
        val secondDepth = DepthFusion.relativeDepthForBox(secondBox, depthMap)

        assertEquals(0.35f, firstDepth!!, 0.0001f)
        assertEquals(0.35f, secondDepth!!, 0.0001f)
    }

    @Test
    fun centerCropReducesBorderDepthNoise() {
        val depthMap = Array(10) { FloatArray(10) { 9.0f } }
        for (row in 2 until 8) {
            for (col in 2 until 8) {
                depthMap[row][col] = 0.42f
            }
        }
        val box = YoloDetectionBox(classId = 60, score = 0.9f, cx = 0.5f, cy = 0.5f, w = 1.0f, h = 1.0f)

        val depth = DepthFusion.relativeDepthForBox(box, depthMap)

        assertEquals(0.42f, depth!!, 0.0001f)
    }
}
