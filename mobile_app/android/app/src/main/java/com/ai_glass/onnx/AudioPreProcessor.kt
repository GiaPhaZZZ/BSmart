package com.ai_glass.onnx

import kotlin.math.*

/**
 * Log-Mel Spectrogram preprocessor for PhoWhisper (Whisper architecture).
 * Produces output shape [1, 80, 3000] matching phowhisper_encoder.onnx input 'mel_input'.
 *
 * Verified from ONNX inspection:
 *   Input: mel_input [batch_size, 80, mel_time] FLOAT
 *
 * Spec (from model_meta.json):
 *   sample_rate:  16000 Hz
 *   n_fft:        400
 *   hop_length:   160
 *   n_mels:       80
 *   chunk_length: 30s  → 3000 frames
 */
object AudioPreProcessor {
    private const val SAMPLE_RATE = 16000
    private const val N_FFT = 400
    private const val HOP_LENGTH = 160
    private const val N_MELS = 80
    private const val TARGET_FRAMES = 3000  // 30s * 100 frames/s
    private const val F_MIN = 0.0
    private const val F_MAX = 8000.0

    // Mel filterbank — computed once and cached
    private val melFilterbank: Array<FloatArray> by lazy { buildMelFilterbank() }

    /**
     * Compute Log-Mel Spectrogram from raw 16-bit PCM samples.
     * Returns FloatArray of size N_MELS * TARGET_FRAMES (for flat tensor).
     */
    fun computeLogMelSpectrogram(pcmData: ShortArray): FloatArray {
        // 1. Normalize PCM to float [-1.0, 1.0]
        val audioLength = SAMPLE_RATE * 30  // pad/clip to 30s
        val audio = FloatArray(audioLength)
        for (i in pcmData.indices) {
            if (i < audioLength) {
                audio[i] = pcmData[i].toFloat() / 32768.0f
            }
        }

        // 2. Build Hann window
        val window = FloatArray(N_FFT) { i ->
            (0.5f * (1.0f - cos(2.0 * PI * i / (N_FFT - 1)))).toFloat()
        }

        // 3. STFT — real DFT for each frame
        // Number of real frequency bins: N_FFT/2 + 1 = 201
        val numFreqBins = N_FFT / 2 + 1
        val powerSpectrum = Array(TARGET_FRAMES) { FloatArray(numFreqBins) }

        for (t in 0 until TARGET_FRAMES) {
            val start = t * HOP_LENGTH
            // Windowed frame
            val frame = FloatArray(N_FFT)
            for (i in 0 until N_FFT) {
                val srcIdx = start + i
                frame[i] = if (srcIdx < audio.size) audio[srcIdx] * window[i] else 0f
            }
            // Compute power spectrum via DFT (real-valued input)
            for (k in 0 until numFreqBins) {
                var re = 0.0
                var im = 0.0
                val factor = 2.0 * PI * k / N_FFT
                for (n in 0 until N_FFT) {
                    re += frame[n] * cos(factor * n)
                    im -= frame[n] * sin(factor * n)
                }
                powerSpectrum[t][k] = (re * re + im * im).toFloat()
            }
        }

        // 4. Apply Mel filterbank → [TARGET_FRAMES, N_MELS]
        val filters = melFilterbank
        val melSpec = Array(TARGET_FRAMES) { FloatArray(N_MELS) }
        for (t in 0 until TARGET_FRAMES) {
            for (m in 0 until N_MELS) {
                var sum = 0f
                for (k in 0 until numFreqBins) {
                    sum += filters[m][k] * powerSpectrum[t][k]
                }
                melSpec[t][m] = sum
            }
        }

        // 5. Log compression + Whisper normalization
        // log10(max(mel, 1e-10)), clamp to max-8, then +4 and /4
        var globalMax = Float.NEGATIVE_INFINITY
        val logMel = Array(TARGET_FRAMES) { t ->
            FloatArray(N_MELS) { m ->
                val v = log10(max(melSpec[t][m], 1e-10f))
                if (v > globalMax) globalMax = v
                v
            }
        }

        val threshold = globalMax - 8.0f
        val result = FloatArray(N_MELS * TARGET_FRAMES)
        // Whisper output layout: [N_MELS, TARGET_FRAMES] (mel-major)
        for (m in 0 until N_MELS) {
            for (t in 0 until TARGET_FRAMES) {
                val v = max(logMel[t][m], threshold)
                result[m * TARGET_FRAMES + t] = (v + 4.0f) / 4.0f
            }
        }

        return result
    }

    /**
     * Build triangular Mel filterbank of shape [N_MELS, numFreqBins].
     * Uses HTK-style Mel scale: mel = 2595 * log10(1 + f / 700).
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val numFreqBins = N_FFT / 2 + 1
        val freqBinHz = DoubleArray(numFreqBins) { k -> k.toDouble() * SAMPLE_RATE / N_FFT }

        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

        val melMin = hzToMel(F_MIN)
        val melMax = hzToMel(F_MAX)
        // N_MELS + 2 mel center points (including boundaries)
        val melPoints = DoubleArray(N_MELS + 2) { i ->
            melMin + i.toDouble() * (melMax - melMin) / (N_MELS + 1)
        }
        val hzPoints = DoubleArray(N_MELS + 2) { melToHz(melPoints[it]) }

        val filters = Array(N_MELS) { FloatArray(numFreqBins) }
        for (m in 0 until N_MELS) {
            val fLeft   = hzPoints[m]
            val fCenter = hzPoints[m + 1]
            val fRight  = hzPoints[m + 2]
            for (k in 0 until numFreqBins) {
                val fk = freqBinHz[k]
                filters[m][k] = when {
                    fk < fLeft  -> 0f
                    fk < fCenter -> ((fk - fLeft) / (fCenter - fLeft)).toFloat()
                    fk < fRight  -> ((fRight - fk) / (fRight - fCenter)).toFloat()
                    else          -> 0f
                }
            }
        }
        return filters
    }
}
