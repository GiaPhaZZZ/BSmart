package com.ai_glass.onnx

import android.content.Context
import org.json.JSONArray

/**
 * PhoWhisper detokenizer sử dụng vocabulary array thật từ models/phowhisper/vocab.json.
 * vocab.json là JSON array: index = token_id, value = token_string.
 * Vocab size: 51,865 tokens (Whisper vocabulary).
 *
 * Verified from: d:\BSmart\models\phowhisper-ct2-int8\vocabulary.json (51,865 tokens).
 * ONNX output vocab_size: 51,865 (confirmed from phowhisper_decoder.onnx inspection).
 */
class PhoWhisperDetokenizer(context: Context) {
    private val idToToken = mutableMapOf<Int, String>()

    init {
        try {
            val assetManager = context.assets
            val inputStream = assetManager.open("models/phowhisper/vocab.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            inputStream.close()
            // vocab.json is a JSON Array: index = token_id
            val jsonArray = JSONArray(jsonString)
            for (i in 0 until jsonArray.length()) {
                idToToken[i] = jsonArray.getString(i)
            }
            android.util.Log.i("PhoWhisperDetokenizer", "Loaded ${idToToken.size} tokens from vocab.json")
        } catch (e: Exception) {
            android.util.Log.e("PhoWhisperDetokenizer", "Failed to load vocab.json: ${e.message}", e)
            // Do NOT populate with fake entries. If vocab fails to load, decode() will return empty.
        }
    }

    /**
     * Returns true if vocabulary was loaded successfully.
     */
    fun isReady(): Boolean = idToToken.size > 100

    /**
     * Decode token IDs to Vietnamese text.
     * Skips special tokens (wrapped in <|...|>).
     * Handles Whisper's byte-level encoding (Ġ prefix = space character U+0120).
     */
    fun decode(tokenIds: IntArray): String {
        if (!isReady()) {
            throw IllegalStateException("Vocabulary not loaded — cannot decode tokens")
        }
        val sb = StringBuilder()
        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            // Skip special control tokens like <|startoftranscript|>, <|vi|>, etc.
            if (token.startsWith("<|") && token.endsWith("|>")) continue
            // Whisper uses byte-level BPE: Ġ (U+0120) represents a leading space
            val cleaned = token.replace('\u0120', ' ')
            sb.append(cleaned)
        }
        return sb.toString().trim()
    }
}
