package com.ai_glass.onnx

import android.content.Context

/**
 * GPT-style byte-level BPE tokenizer for SmolVLM2.
 *
 * The base vocabulary comes from vocab.json. SmolVLM2 image/layout tokens live in
 * added_tokens.json in the HF snapshot, but only vocab.json/merges.txt are bundled as
 * Android assets, so the official added token ids are registered here.
 */
class SmolVLM2Tokenizer(private val context: Context) {

    companion object {
        const val EOS_TOKEN_ID = 49279
        const val BOS_TOKEN_ID = 1
        const val GLOBAL_IMAGE_TOKEN_ID = 49152
        const val FAKE_IMAGE_TOKEN_ID = 49189
        const val IMAGE_TOKEN_ID = 49190

        private val SPECIAL_TOKEN_IDS: Map<String, Int> by lazy {
            buildMap {
                put("<|endoftext|>", 0)
                put("<|im_start|>", BOS_TOKEN_ID)
                put("<|im_end|>", 2)
                put("<global-img>", GLOBAL_IMAGE_TOKEN_ID)
                for (row in 1..6) {
                    for (col in 1..6) {
                        put("<row_${row}_col_${col}>", GLOBAL_IMAGE_TOKEN_ID + ((row - 1) * 6) + col)
                    }
                }
                put("<fake_token_around_image>", FAKE_IMAGE_TOKEN_ID)
                put("<image>", IMAGE_TOKEN_ID)
                put("<end_of_utterance>", EOS_TOKEN_ID)
            }
        }

        private val SPECIAL_TOKENS_BY_LENGTH: List<String> by lazy {
            SPECIAL_TOKEN_IDS.keys.sortedByDescending { it.length }
        }

        private val BYTES_TO_UNICODE: Map<Int, Char> by lazy {
            val bs = mutableListOf<Int>()
            bs.addAll('!'.code..'~'.code)
            bs.addAll(0x00A1..0x00AC)
            bs.addAll(0x00AE..0x00FF)

            val cs = bs.map { it.toChar() }.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add((256 + n).toChar())
                    n++
                }
            }
            bs.zip(cs).toMap()
        }

        private val UNICODE_TO_BYTES: Map<Char, Int> by lazy {
            BYTES_TO_UNICODE.entries.associate { (k, v) -> v to k }
        }
    }

    private val encoder = mutableMapOf<String, Int>()
    private val decoder = mutableMapOf<Int, String>()
    private val bpeRanks = mutableMapOf<Pair<String, String>, Int>()

    private var loaded = false

    init {
        try {
            loadVocab()
            loadMerges()
            loaded = true
            android.util.Log.i("SmolVLM2Tokenizer", "Loaded ${encoder.size} tokens, ${bpeRanks.size} BPE merges")
        } catch (e: Exception) {
            android.util.Log.e("SmolVLM2Tokenizer", "Failed to load tokenizer: ${e.message}", e)
        }
    }

    fun isReady(): Boolean = loaded && encoder.isNotEmpty() && bpeRanks.isNotEmpty()

    private fun loadVocab() {
        val inputStream = context.assets.open("models/smolvlm2/vocab.json")
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        inputStream.close()

        val pattern = Regex(""""((?:[^"\\]|\\.)*)"\s*:\s*(\d+)""")
        pattern.findAll(jsonString).forEach { match ->
            val token = match.groupValues[1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\/", "/")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
            val id = match.groupValues[2].toIntOrNull() ?: return@forEach
            encoder[token] = id
            decoder[id] = token
        }

        for ((token, id) in SPECIAL_TOKEN_IDS) {
            encoder[token] = id
            decoder[id] = token
        }
    }

    private fun loadMerges() {
        val inputStream = context.assets.open("models/smolvlm2/merges.txt")
        val lines = inputStream.bufferedReader().use { it.readLines() }
        inputStream.close()

        var rank = 0
        for (line in lines) {
            if (line.startsWith("#version") || line.isBlank()) continue
            val parts = line.split(" ")
            if (parts.size == 2) {
                bpeRanks[Pair(parts[0], parts[1])] = rank++
            }
        }
    }

    private fun bpe(token: String): List<String> {
        var word = token.map { it.toString() }.toMutableList()

        while (word.size > 1) {
            var minRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null
            for (i in 0 until word.size - 1) {
                val pair = Pair(word[i], word[i + 1])
                val rank = bpeRanks[pair] ?: Int.MAX_VALUE
                if (rank < minRank) {
                    minRank = rank
                    bestPair = pair
                }
            }
            if (bestPair == null || minRank == Int.MAX_VALUE) break

            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == bestPair.first && word[i + 1] == bestPair.second) {
                    newWord.add(bestPair.first + bestPair.second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i++
                }
            }
            word = newWord
        }
        return word
    }

    fun encode(text: String): IntArray {
        if (!isReady()) throw IllegalStateException("Tokenizer not loaded")

        val ids = mutableListOf<Int>()
        val plainText = StringBuilder()

        fun flushPlainText() {
            if (plainText.isEmpty()) return
            ids.addAll(encodePlainText(plainText.toString()).toList())
            plainText.clear()
        }

        var index = 0
        while (index < text.length) {
            val specialToken = SPECIAL_TOKENS_BY_LENGTH.firstOrNull { text.startsWith(it, index) }
            if (specialToken != null) {
                flushPlainText()
                ids.add(SPECIAL_TOKEN_IDS.getValue(specialToken))
                index += specialToken.length
            } else {
                plainText.append(text[index])
                index++
            }
        }
        flushPlainText()
        return ids.toIntArray()
    }

    private fun encodePlainText(text: String): IntArray {
        val ids = mutableListOf<Int>()
        val regex = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        for (match in regex.findAll(text)) {
            val byteEncoded = StringBuilder()
            for (byte in match.value.toByteArray(Charsets.UTF_8)) {
                val ch = BYTES_TO_UNICODE[byte.toInt() and 0xFF] ?: continue
                byteEncoded.append(ch)
            }

            for (bpeToken in bpe(byteEncoded.toString())) {
                val id = encoder[bpeToken]
                if (id != null) {
                    ids.add(id)
                } else {
                    for (ch in bpeToken) {
                        val byteId = encoder[ch.toString()]
                        if (byteId != null) ids.add(byteId)
                    }
                }
            }
        }

        return ids.toIntArray()
    }

    fun decode(tokenIds: IntArray): String {
        if (!isReady()) throw IllegalStateException("Tokenizer not loaded")

        val tokens = tokenIds
            .filterNot { id ->
                id == EOS_TOKEN_ID ||
                    id == BOS_TOKEN_ID ||
                    id == FAKE_IMAGE_TOKEN_ID ||
                    id == IMAGE_TOKEN_ID ||
                    id in GLOBAL_IMAGE_TOKEN_ID..49188
            }
            .map { id -> decoder[id] }
            .filterNotNull()
        val text = tokens.joinToString("")

        val bytes = mutableListOf<Byte>()
        for (ch in text) {
            val b = UNICODE_TO_BYTES[ch]
            if (b != null) {
                bytes.add(b.toByte())
            }
        }

        return try {
            String(bytes.toByteArray(), Charsets.UTF_8)
        } catch (e: Exception) {
            tokens.joinToString("").replace("\u0120", " ").trim()
        }
    }
}
