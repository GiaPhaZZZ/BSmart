#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

namespace {

constexpr const char * TAG = "BsmartLlamaVlm";
constexpr int DEFAULT_BATCH_SIZE = 512;
constexpr const char * DEFAULT_MODEL_NAME = "SmolVLM2-256M-Video-Instruct-GGUF";

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u";
                    const char * hex = "0123456789abcdef";
                    out << "00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

jstring to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return "";
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

uint64_t file_size(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return 0;
    }
    return static_cast<uint64_t>(in.tellg());
}

bool file_exists(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

std::string error_json(const std::string & code, const std::string & message) {
    return std::string("{\"isSuccess\":false,\"runtimeAvailable\":") +
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
        "true"
#else
        "false"
#endif
        + ",\"error\":\"" + json_escape(code) + "\",\"message\":\"" + json_escape(message) + "\"}";
}

#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)

struct NativeState {
    bool backend_initialized = false;
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
    llama_sampler * sampler = nullptr;
    llama_batch batch{};
    bool batch_ready = false;
    std::string model_path;
    std::string mmproj_path;
    std::string model_desc = DEFAULT_MODEL_NAME;
    int context_size = 2048;
    int threads = 2;
    int max_tokens = 48;
};

NativeState g_state;
std::mutex g_mutex;
int g_image_max_tokens_override = 512;

void clear_batch(llama_batch & batch) {
    batch.n_tokens = 0;
}

void add_token_to_batch(llama_batch & batch, llama_token token, llama_pos pos, bool logits) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = logits ? 1 : 0;
}

void free_loaded_state_locked() {
    if (g_state.sampler != nullptr) {
        llama_sampler_free(g_state.sampler);
        g_state.sampler = nullptr;
    }
    if (g_state.batch_ready) {
        llama_batch_free(g_state.batch);
        g_state.batch_ready = false;
    }
    if (g_state.mctx != nullptr) {
        mtmd_free(g_state.mctx);
        g_state.mctx = nullptr;
    }
    if (g_state.lctx != nullptr) {
        llama_free(g_state.lctx);
        g_state.lctx = nullptr;
    }
    if (g_state.model != nullptr) {
        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }
    g_state.model_path.clear();
    g_state.mmproj_path.clear();
    g_state.model_desc = DEFAULT_MODEL_NAME;
}

std::string build_prompt(const std::string & user_prompt) {
    std::string concise = user_prompt;
    if (concise.find("Answer") == std::string::npos && concise.find("answer") == std::string::npos) {
        concise += "\nAnswer in 1 or 2 short sentences for a blind user.";
    }
    return std::string("<|im_start|>User:") + mtmd_default_marker() + "\n" +
           concise + "<end_of_utterance>\nAssistant:";
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::vector<char> text(tokens.size() * 16 + 64);
    int32_t written = llama_detokenize(
            vocab,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            text.data(),
            static_cast<int32_t>(text.size()),
            true,
            false);
    if (written < 0) {
        text.resize(static_cast<size_t>(-written) + 1);
        written = llama_detokenize(
                vocab,
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                text.data(),
                static_cast<int32_t>(text.size()),
                true,
                false);
    }
    if (written <= 0) {
        return "";
    }
    return std::string(text.data(), static_cast<size_t>(written));
}

std::vector<std::string> split_on_marker(const std::string & prompt) {
    const std::string marker = mtmd_default_marker();
    std::vector<std::string> segments;
    size_t start = 0;
    size_t pos = prompt.find(marker, start);
    while (pos != std::string::npos) {
        segments.push_back(prompt.substr(start, pos - start));
        start = pos + marker.size();
        pos = prompt.find(marker, start);
    }
    segments.push_back(prompt.substr(start));
    return segments;
}

std::string run_visual_qa_locked(
        const std::string & image_path,
        const std::string & prompt,
        int max_tokens,
        double temperature) {
    if (g_state.model == nullptr || g_state.lctx == nullptr || g_state.mctx == nullptr) {
        return error_json("MODEL_NOT_LOADED", "SmolVLM2 GGUF is not loaded");
    }
    if (!file_exists(image_path)) {
        return error_json("IMAGE_NOT_FOUND", "Image path does not exist: " + image_path);
    }
    if (prompt.empty()) {
        return error_json("EMPTY_PROMPT", "Prompt is empty");
    }

    max_tokens = max_tokens > 0 ? max_tokens : g_state.max_tokens;
    max_tokens = std::max(1, std::min(max_tokens, 128));

    const int64_t total_start = now_ms();
    __android_log_print(ANDROID_LOG_INFO, TAG, "runVisualQA start image=%s max_tokens=%d", image_path.c_str(), max_tokens);
    int64_t image_preprocessing_ms = 0;
    int64_t vision_encoding_ms = 0;
    int64_t prompt_processing_ms = 0;
    int64_t generation_ms = 0;
    int64_t time_to_first_token_ms = -1;
    llama_pos n_past = 0;

    llama_memory_clear(llama_get_memory(g_state.lctx), true);
    if (g_state.sampler != nullptr) {
        llama_sampler_reset(g_state.sampler);
    }

    std::string detected_format = "UNKNOWN";
    {
        std::ifstream img_file(image_path, std::ios::binary);
        if (img_file) {
            unsigned char header[12] = {0};
            img_file.read(reinterpret_cast<char *>(header), sizeof(header));
            const std::streamsize bytes_read = img_file.gcount();
            if (bytes_read >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF) {
                detected_format = "image/jpeg";
            } else if (bytes_read >= 8 && header[0] == 0x89 && header[1] == 0x4E && header[2] == 0x47 && header[3] == 0x0D) {
                // Actually PNG signature is 89 50 4E 47 0D 0A 1A 0A
                detected_format = "image/png";
            } else if (bytes_read >= 8 && header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47) {
                detected_format = "image/png";
            } else if (bytes_read >= 12 && header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F' &&
                       header[8] == 'W' && header[9] == 'E' && header[10] == 'B' && header[11] == 'P') {
                detected_format = "image/webp";
            }
        }
    }

    const int64_t image_start = now_ms();
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    mtmd_helper_bitmap_wrapper bitmap_wrapper =
            mtmd_helper_bitmap_init_from_file(g_state.mctx, image_path.c_str(), false, init_opt);
    image_preprocessing_ms = now_ms() - image_start;
    __android_log_print(ANDROID_LOG_INFO, TAG, "image preprocessing done in %lld ms", static_cast<long long>(image_preprocessing_ms));
    if (bitmap_wrapper.bitmap == nullptr) {
        if (bitmap_wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(bitmap_wrapper.video_ctx);
        }
        __android_log_print(ANDROID_LOG_ERROR, TAG, "IMAGE_PROOF format=%s valid=false decode_failed=true", detected_format.c_str());
        return error_json("INVALID_IMAGE", "libmtmd could not decode image: " + image_path + " format: " + detected_format);
    }

    const uint32_t img_width = mtmd_bitmap_get_nx(bitmap_wrapper.bitmap);
    const uint32_t img_height = mtmd_bitmap_get_ny(bitmap_wrapper.bitmap);
    const size_t decoded_byte_count = mtmd_bitmap_get_n_bytes(bitmap_wrapper.bitmap);
    const size_t expected_byte_count = static_cast<size_t>(img_width) * img_height * 3;
    const bool is_buffer_valid = (decoded_byte_count == expected_byte_count && decoded_byte_count > 0 && mtmd_bitmap_get_data(bitmap_wrapper.bitmap) != nullptr);

    __android_log_print(ANDROID_LOG_INFO, TAG,
        "IMAGE_PROOF format=%s width=%u height=%u channels=3 decoded_bytes=%zu expected_bytes=%zu valid=%s",
        detected_format.c_str(), img_width, img_height, decoded_byte_count, expected_byte_count,
        (is_buffer_valid ? "true" : "false"));

    if (!is_buffer_valid) {
        mtmd_bitmap_free(bitmap_wrapper.bitmap);
        if (bitmap_wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(bitmap_wrapper.video_ctx);
        }
        return error_json("INVALID_IMAGE_BUFFER", "Decoded bitmap bytes mismatch expected RGB24 size");
    }

    const std::string formatted_prompt = build_prompt(prompt);
    std::vector<std::string> segments = split_on_marker(formatted_prompt);
    if (segments.size() != 2) {
        mtmd_bitmap_free(bitmap_wrapper.bitmap);
        if (bitmap_wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(bitmap_wrapper.video_ctx);
        }
        return error_json("PROMPT_MARKER_ERROR", "Expected exactly one multimodal marker in prompt");
    }

    std::vector<mtmd_input_text> texts(segments.size());
    std::vector<mtmd_input_part> parts;
    for (size_t i = 0; i < segments.size(); ++i) {
        texts[i] = {segments[i].data(), segments[i].size(), false, true};
        parts.push_back({&texts[i], nullptr});
        if (i == 0) {
            parts.push_back({nullptr, bitmap_wrapper.bitmap});
        }
    }

    std::vector<const mtmd_input_part *> part_ptrs;
    for (const auto & part : parts) {
        part_ptrs.push_back(&part);
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int32_t tokenize_result = mtmd_tokenize_from_parts(
            g_state.mctx,
            chunks,
            part_ptrs.data(),
            part_ptrs.size(),
            true);
    mtmd_bitmap_free(bitmap_wrapper.bitmap);
    if (bitmap_wrapper.video_ctx != nullptr) {
        mtmd_helper_video_free(bitmap_wrapper.video_ctx);
    }
    if (tokenize_result != 0) {
        mtmd_input_chunks_free(chunks);
        return error_json("TOKENIZE_FAILED", "libmtmd tokenize failed with code " + std::to_string(tokenize_result));
    }

    const size_t chunk_count = mtmd_input_chunks_size(chunks);
    size_t text_chunk_count = 0;
    size_t vision_chunk_count = 0;
    size_t total_image_tokens = 0;
    for (size_t i = 0; i < chunk_count; ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            text_chunk_count++;
        } else {
            vision_chunk_count++;
            total_image_tokens += mtmd_input_chunk_get_n_tokens(c);
        }
    }
    __android_log_print(ANDROID_LOG_INFO, TAG,
        "VISION_WORKLOAD total_chunks=%zu text_chunks=%zu vision_chunks=%zu total_image_tokens=%zu",
        chunk_count, text_chunk_count, vision_chunk_count, total_image_tokens);
    for (size_t i = 0; i < chunk_count; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const auto chunk_type = mtmd_input_chunk_get_type(chunk);
        __android_log_print(ANDROID_LOG_INFO, TAG, "eval chunk %zu/%zu type=%d", i + 1, chunk_count, static_cast<int>(chunk_type));
        const bool logits_last = (i == chunk_count - 1);
        llama_pos new_n_past = n_past;

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            const int64_t prompt_start = now_ms();
            int32_t eval_result = mtmd_helper_eval_chunk_single(
                    g_state.mctx,
                    g_state.lctx,
                    chunk,
                    n_past,
                    0,
                    DEFAULT_BATCH_SIZE,
                    logits_last,
                    &new_n_past);
            prompt_processing_ms += now_ms() - prompt_start;
            __android_log_print(ANDROID_LOG_INFO, TAG, "text eval chunk %zu done n_past=%d", i + 1, static_cast<int>(new_n_past));
            if (eval_result != 0) {
                mtmd_input_chunks_free(chunks);
                return error_json("TEXT_EVAL_FAILED", "llama text eval failed with code " + std::to_string(eval_result));
            }
            n_past = new_n_past;
        } else {
            const int64_t vision_start = now_ms();
            __android_log_print(ANDROID_LOG_INFO, TAG, "vision encode chunk %zu start", i + 1);
            int32_t encode_result = mtmd_encode_chunk(g_state.mctx, chunk);
            vision_encoding_ms += now_ms() - vision_start;
            __android_log_print(ANDROID_LOG_INFO, TAG, "vision encode chunk %zu done total_vision_ms=%lld", i + 1, static_cast<long long>(vision_encoding_ms));
            if (encode_result != 0) {
                mtmd_input_chunks_free(chunks);
                return error_json("VISION_ENCODE_FAILED", "libmtmd image encode failed with code " + std::to_string(encode_result));
            }
            float * embd = mtmd_get_output_embd(g_state.mctx);
            if (embd == nullptr) {
                mtmd_input_chunks_free(chunks);
                return error_json("VISION_EMBEDDING_MISSING", "libmtmd returned null image embeddings");
            }

            const int64_t prompt_start = now_ms();
            __android_log_print(ANDROID_LOG_INFO, TAG, "image decode chunk %zu start n_past=%d", i + 1, static_cast<int>(n_past));
            int32_t decode_result = mtmd_helper_decode_image_chunk(
                    g_state.mctx,
                    g_state.lctx,
                    chunk,
                    embd,
                    n_past,
                    0,
                    DEFAULT_BATCH_SIZE,
                    &new_n_past,
                    nullptr,
                    nullptr);
            prompt_processing_ms += now_ms() - prompt_start;
            __android_log_print(ANDROID_LOG_INFO, TAG, "image decode chunk %zu done n_past=%d", i + 1, static_cast<int>(new_n_past));
            if (decode_result != 0) {
                mtmd_input_chunks_free(chunks);
                return error_json("IMAGE_DECODE_FAILED", "llama image embedding decode failed with code " + std::to_string(decode_result));
            }
            n_past = new_n_past;
        }
    }
    mtmd_input_chunks_free(chunks);

    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(static_cast<size_t>(max_tokens));
    const int64_t generation_start = now_ms();
    __android_log_print(ANDROID_LOG_INFO, TAG, "generation start max_tokens=%d n_past=%d", max_tokens, static_cast<int>(n_past));
    for (int i = 0; i < max_tokens; ++i) {
        llama_token token = llama_sampler_sample(g_state.sampler, g_state.lctx, -1);
        if (time_to_first_token_ms < 0) {
            time_to_first_token_ms = now_ms() - generation_start;
        }
        llama_sampler_accept(g_state.sampler, token);

        if (llama_vocab_is_eog(llama_model_get_vocab(g_state.model), token)) {
            break;
        }

        generated_tokens.push_back(token);
        __android_log_print(ANDROID_LOG_INFO, TAG, "generated token %d/%d", i + 1, max_tokens);
        clear_batch(g_state.batch);
        add_token_to_batch(g_state.batch, token, n_past, true);
        const int32_t decode_result = llama_decode(g_state.lctx, g_state.batch);
        if (decode_result != 0) {
            generation_ms = now_ms() - generation_start;
            return error_json("GENERATION_DECODE_FAILED", "llama generation decode failed with code " + std::to_string(decode_result));
        }
        ++n_past;
    }
    generation_ms = now_ms() - generation_start;
    __android_log_print(ANDROID_LOG_INFO, TAG, "generation done tokens=%zu generation_ms=%lld", generated_tokens.size(), static_cast<long long>(generation_ms));

    const std::string answer = detokenize(llama_model_get_vocab(g_state.model), generated_tokens);
    const int64_t total_ms = now_ms() - total_start;
    const double tokens_per_second = generation_ms > 0
            ? static_cast<double>(generated_tokens.size()) * 1000.0 / static_cast<double>(generation_ms)
            : 0.0;

    std::ostringstream out;
    out << "{\"isSuccess\":true"
        << ",\"answer\":\"" << json_escape(answer) << "\""
        << ",\"model\":\"" << json_escape(g_state.model_desc) << "\""
        << ",\"runtime\":\"llama.cpp/libmtmd\""
        << ",\"quantization\":\"GGUF\""
        << ",\"maxTokens\":" << max_tokens
        << ",\"generatedTokens\":" << generated_tokens.size()
        << ",\"tokensPerSecond\":" << tokens_per_second
        << ",\"timeToFirstTokenMs\":" << time_to_first_token_ms
        << ",\"timings\":{"
        << "\"image_preprocessing_ms\":" << image_preprocessing_ms
        << ",\"vision_encoding_ms\":" << vision_encoding_ms
        << ",\"prompt_processing_ms\":" << prompt_processing_ms
        << ",\"generation_ms\":" << generation_ms
        << ",\"total_vlm_ms\":" << total_ms
        << "}}";
    return out.str();
}

#endif

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ai_1glass_llama_LlamaCppVlmNative_getStatusNative(JNIEnv * env, jobject /*thiz*/) {
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ostringstream out;
    out << "{\"nativeLibraryLoaded\":true"
        << ",\"runtimeAvailable\":true"
        << ",\"compiledWithLlamaCpp\":true"
        << ",\"modelLoaded\":" << (g_state.model != nullptr ? "true" : "false")
        << ",\"runtime\":\"llama.cpp/libmtmd\""
        << ",\"model\":\"" << json_escape(g_state.model_desc) << "\""
        << ",\"modelPath\":\"" << json_escape(g_state.model_path) << "\""
        << ",\"mmprojPath\":\"" << json_escape(g_state.mmproj_path) << "\""
        << ",\"contextSize\":" << g_state.context_size
        << ",\"threads\":" << g_state.threads
        << ",\"maxTokens\":" << g_state.max_tokens;
    if (g_state.model != nullptr) {
        out << ",\"modelTensorBytes\":" << llama_model_size(g_state.model)
            << ",\"modelParams\":" << llama_model_n_params(g_state.model)
            << ",\"visionSupported\":" << (mtmd_support_vision(g_state.mctx) ? "true" : "false");
    }
    out << "}";
    return to_jstring(env, out.str());
#else
    return to_jstring(env,
        "{\"nativeLibraryLoaded\":true,"
        "\"runtimeAvailable\":false,"
        "\"compiledWithLlamaCpp\":false,"
        "\"modelLoaded\":false,"
        "\"runtime\":\"stub\","
        "\"message\":\"Built without llama.cpp/libmtmd. Rebuild with -PbsmartLlamaWithLlamaCpp=true and -PbsmartLlamaCppDir=<path>.\"}");
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ai_1glass_llama_LlamaCppVlmNative_loadModelNative(
        JNIEnv * env,
        jobject /*thiz*/,
        jstring jmodel_path,
        jstring jmmproj_path,
        jint context_size,
        jint threads,
        jint max_tokens,
        jdouble temperature) {
    (void) temperature;
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string model_path = jstring_to_string(env, jmodel_path);
    const std::string mmproj_path = jstring_to_string(env, jmmproj_path);
    if (!file_exists(model_path)) {
        return to_jstring(env, error_json("MODEL_NOT_FOUND", "GGUF model not found: " + model_path));
    }
    if (!file_exists(mmproj_path)) {
        return to_jstring(env, error_json("MMPROJ_NOT_FOUND", "GGUF mmproj not found: " + mmproj_path));
    }

    free_loaded_state_locked();
    const int64_t started = now_ms();
    if (!g_state.backend_initialized) {
        llama_backend_init();
        g_state.backend_initialized = true;
    }

    llama_model_params model_params = llama_model_default_params();
    g_state.model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (g_state.model == nullptr) {
        free_loaded_state_locked();
        return to_jstring(env, error_json("MODEL_LOAD_FAILED", "llama_model_load_from_file returned null"));
    }

    g_state.context_size = context_size > 0 ? context_size : 2048;
    g_state.threads = threads > 0 ? threads : 2;
    g_state.max_tokens = max_tokens > 0 ? max_tokens : 48;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(g_state.context_size);
    ctx_params.n_batch = DEFAULT_BATCH_SIZE;
    ctx_params.n_ubatch = DEFAULT_BATCH_SIZE;
    ctx_params.n_threads = g_state.threads;
    ctx_params.n_threads_batch = g_state.threads;
    g_state.lctx = llama_init_from_model(g_state.model, ctx_params);
    if (g_state.lctx == nullptr) {
        free_loaded_state_locked();
        return to_jstring(env, error_json("CONTEXT_INIT_FAILED", "llama_init_from_model returned null"));
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.print_timings = true;
    mtmd_params.n_threads = g_state.threads;
    mtmd_params.warmup = true;
    mtmd_params.image_max_tokens = g_image_max_tokens_override;
    __android_log_print(ANDROID_LOG_INFO, TAG, "loadModelNative image_max_tokens=%d threads=%d", g_image_max_tokens_override, g_state.threads);
    g_state.mctx = mtmd_init_from_file(mmproj_path.c_str(), g_state.model, mtmd_params);
    if (g_state.mctx == nullptr) {
        free_loaded_state_locked();
        return to_jstring(env, error_json("MMPROJ_LOAD_FAILED", "mtmd_init_from_file returned null"));
    }
    if (!mtmd_support_vision(g_state.mctx)) {
        free_loaded_state_locked();
        return to_jstring(env, error_json("MMPROJ_NO_VISION", "Loaded mmproj does not report vision support"));
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    g_state.sampler = llama_sampler_chain_init(sampler_params);
    if (temperature > 0.0) {
        llama_sampler_chain_add(g_state.sampler, llama_sampler_init_temp(static_cast<float>(temperature)));
        llama_sampler_chain_add(g_state.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(g_state.sampler, llama_sampler_init_greedy());
    }
    g_state.batch = llama_batch_init(DEFAULT_BATCH_SIZE, 0, 1);
    g_state.batch_ready = true;
    g_state.model_path = model_path;
    g_state.mmproj_path = mmproj_path;

    char desc[256] = {0};
    if (llama_model_desc(g_state.model, desc, sizeof(desc)) > 0) {
        g_state.model_desc = desc;
    }

    const int64_t load_ms = now_ms() - started;
    std::ostringstream out;
    out << "{\"isSuccess\":true"
        << ",\"runtimeAvailable\":true"
        << ",\"modelLoaded\":true"
        << ",\"runtime\":\"llama.cpp/libmtmd\""
        << ",\"model\":\"" << json_escape(g_state.model_desc) << "\""
        << ",\"modelPath\":\"" << json_escape(g_state.model_path) << "\""
        << ",\"mmprojPath\":\"" << json_escape(g_state.mmproj_path) << "\""
        << ",\"modelFileBytes\":" << file_size(model_path)
        << ",\"mmprojFileBytes\":" << file_size(mmproj_path)
        << ",\"modelTensorBytes\":" << llama_model_size(g_state.model)
        << ",\"modelParams\":" << llama_model_n_params(g_state.model)
        << ",\"contextSize\":" << llama_n_ctx(g_state.lctx)
        << ",\"threads\":" << g_state.threads
        << ",\"maxTokens\":" << g_state.max_tokens
        << ",\"timings\":{\"model_load_ms\":" << load_ms << "}}";
    return to_jstring(env, out.str());
#else
    (void) env;
    (void) jmodel_path;
    (void) jmmproj_path;
    (void) context_size;
    (void) threads;
    (void) max_tokens;
    return to_jstring(env, error_json("RUNTIME_NOT_COMPILED", "Native library was built without llama.cpp/libmtmd"));
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ai_1glass_llama_LlamaCppVlmNative_runVisualQANative(
        JNIEnv * env,
        jobject /*thiz*/,
        jstring jimage_path,
        jstring jprompt,
        jint max_tokens,
        jdouble temperature) {
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
    std::lock_guard<std::mutex> lock(g_mutex);
    return to_jstring(env, run_visual_qa_locked(
            jstring_to_string(env, jimage_path),
            jstring_to_string(env, jprompt),
            max_tokens,
            temperature));
#else
    (void) jimage_path;
    (void) jprompt;
    (void) max_tokens;
    (void) temperature;
    return to_jstring(env, error_json("RUNTIME_NOT_COMPILED", "Native library was built without llama.cpp/libmtmd"));
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ai_1glass_llama_LlamaCppVlmNative_unloadModelNative(JNIEnv * env, jobject /*thiz*/) {
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
    std::lock_guard<std::mutex> lock(g_mutex);
    free_loaded_state_locked();
    return to_jstring(env, "{\"isSuccess\":true,\"modelLoaded\":false}");
#else
    return to_jstring(env, "{\"isSuccess\":true,\"modelLoaded\":false,\"runtimeAvailable\":false}");
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_ai_1glass_llama_LlamaCppVlmNative_setImageMaxTokensNative(JNIEnv * /*env*/, jobject /*thiz*/, jint tokens) {
#if defined(BSMART_LLAMA_WITH_LLAMA_CPP)
    g_image_max_tokens_override = tokens > 0 ? tokens : 512;
    __android_log_print(ANDROID_LOG_INFO, TAG, "setImageMaxTokensNative set to %d", g_image_max_tokens_override);
#else
    (void) tokens;
#endif
}

