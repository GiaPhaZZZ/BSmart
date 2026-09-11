package com.ai_glass.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Base64
import android.util.Log
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

/**
 * PhoneCameraModule:
 * Native Android Camera2 module for capturing single photo frames directly from the phone's
 * rear camera without requiring a full-screen preview.
 *
 * Used as an autonomous visual sensor when smart glasses are disconnected or camera is absent.
 */
class PhoneCameraModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val TAG = "PhoneCameraModule"
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    override fun getName(): String = "PhoneCameraModule"

    private fun startBackgroundThread() {
        if (backgroundThread == null) {
            backgroundThread = HandlerThread("PhoneCameraBackground").apply { start() }
            backgroundHandler = Handler(backgroundThread!!.looper)
        }
    }

    private fun stopBackgroundThread() {
        try {
            backgroundThread?.quitSafely()
            backgroundThread?.join(500)
            backgroundThread = null
            backgroundHandler = null
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping background thread", e)
        }
    }

    private fun closeCameraResources() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader?.close()
            imageReader = null
        } catch (e: Exception) {
            Log.w(TAG, "Error closing camera resources", e)
        } finally {
            stopBackgroundThread()
        }
    }

    @ReactMethod
    fun isCameraAvailable(promise: Promise) {
        try {
            val cameraManager = reactContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val hasCamera = cameraManager.cameraIdList.isNotEmpty()
            promise.resolve(hasCamera)
        } catch (e: Exception) {
            promise.resolve(false)
        }
    }

    @SuppressLint("MissingPermission")
    @ReactMethod
    fun capturePhoto(promise: Promise) {
        if (ContextCompat.checkSelfPermission(reactContext, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            promise.reject("ERR_PERMISSION", "Quyền truy cập Camera chưa được cấp")
            return
        }

        startBackgroundThread()

        try {
            val cameraManager = reactContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                val chars = cameraManager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: cameraManager.cameraIdList.firstOrNull()

            if (cameraId == null) {
                closeCameraResources()
                promise.reject("ERR_NO_CAMERA", "Không tìm thấy camera trên điện thoại")
                return
            }

            val chars = cameraManager.getCameraCharacteristics(cameraId)
            val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val jpegSizes = map?.getOutputSizes(ImageFormat.JPEG)
            // Lấy độ phân giải quanh 1080p, hoặc to nhất nếu không có
            var width = 1920
            var height = 1080
            if (jpegSizes != null && jpegSizes.isNotEmpty()) {
                val size = jpegSizes.minByOrNull { Math.abs(it.width * it.height - 1920 * 1080) }
                if (size != null) {
                    width = size.width
                    height = size.height
                }
            }

            imageReader = ImageReader.newInstance(width, height, ImageFormat.JPEG, 2)

            var promiseSettled = false

            imageReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    
                    // Xoay ảnh theo Sensor Orientation
                    val originalBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    val matrix = Matrix()
                    matrix.postRotate(sensorOrientation.toFloat())
                    val rotatedBitmap = Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
                    
                    val outputStream = java.io.ByteArrayOutputStream()
                    rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                    val rotatedBytes = outputStream.toByteArray()
                    val base64 = Base64.encodeToString(rotatedBytes, Base64.NO_WRAP)
                    
                    originalBitmap.recycle()
                    rotatedBitmap.recycle()

                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.resolve(base64)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing captured frame", e)
                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.reject("ERR_PROCESS", e.message)
                    }
                } finally {
                    image.close()
                    closeCameraResources()
                }
            }, backgroundHandler)

            // Safety timeout after 5 seconds if camera hangs
            backgroundHandler?.postDelayed({
                if (!promiseSettled) {
                    promiseSettled = true
                    closeCameraResources()
                    promise.reject("ERR_TIMEOUT", "Thời gian chờ chụp ảnh quá hạn (5s)")
                }
            }, 5000)

            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    try {
                        val surface = imageReader?.surface
                        if (surface == null) {
                            if (!promiseSettled) {
                                promiseSettled = true
                                promise.reject("ERR_SURFACE", "ImageReader surface is null")
                            }
                            closeCameraResources()
                            return
                        }

                        camera.createCaptureSession(
                            listOf(surface),
                            object : CameraCaptureSession.StateCallback() {
                                override fun onConfigured(session: CameraCaptureSession) {
                                    captureSession = session
                                    try {
                                        val captureBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                                            addTarget(surface)
                                            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                                            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                                        }
                                        session.capture(captureBuilder.build(), null, backgroundHandler)
                                    } catch (e: Exception) {
                                        Log.e(TAG, "Capture request failed", e)
                                        if (!promiseSettled) {
                                            promiseSettled = true
                                            promise.reject("ERR_CAPTURE_REQ", e.message)
                                        }
                                        closeCameraResources()
                                    }
                                }

                                override fun onConfigureFailed(session: CameraCaptureSession) {
                                    if (!promiseSettled) {
                                        promiseSettled = true
                                        promise.reject("ERR_CONFIG", "Camera session configuration failed")
                                    }
                                    closeCameraResources()
                                }
                            },
                            backgroundHandler
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to create capture session", e)
                        if (!promiseSettled) {
                            promiseSettled = true
                            promise.reject("ERR_SESSION", e.message)
                        }
                        closeCameraResources()
                    }
                }

                override fun onDisconnected(camera: CameraDevice) {
                    closeCameraResources()
                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.reject("ERR_DISCONNECTED", "Camera disconnected")
                    }
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    closeCameraResources()
                    if (!promiseSettled) {
                        promiseSettled = true
                        promise.reject("ERR_CAMERA", "Camera device error: $error")
                    }
                }
            }, backgroundHandler)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start camera capture", e)
            closeCameraResources()
            promise.reject("ERR_START_CAMERA", e.message)
        }
    }

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        closeCameraResources()
    }
}
