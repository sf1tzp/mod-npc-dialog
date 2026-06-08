#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_httpclient.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include <sstream>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <thread>
#include <mutex>
#include <queue>
#include <future>

std::string ExtractTextBetweenDoubleQuotes(const std::string& response)
{
    // Only strip surrounding double quotes when the entire string is
    // wrapped in them.  The old logic searched for the first two quote
    // characters *anywhere*, which could match quotes inside <think>
    // blocks or adjacent "" pairs and return an empty string.
    if (response.size() >= 2 && response.front() == '"' && response.back() == '"')
        return response.substr(1, response.size() - 2);
    return response;
}

// Function to perform the API call.
std::string QueryOllamaAPI(const std::string& prompt)
{
    // Initialize our custom HTTP client
    static OllamaHttpClient httpClient;
    
    if (!httpClient.IsAvailable())
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: HTTP client not available. Check if Ollama service is running and accessible.");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: HTTP client initialization failed.");
        }
        return "";
    }

    std::string url   = g_OllamaUrl;
    std::string model = g_OllamaModel;

    // Select the wire format. "openai" speaks the OpenAI-compatible
    // /v1/chat/completions API served by LM Studio (and Ollama's own /v1
    // shim, vLLM, llama.cpp, etc.); anything else uses Ollama's native
    // /api/generate endpoint.
    bool useOpenAi = (g_OllamaApiType == "openai");

    // Sanitize the prompt to ensure it's valid UTF-8 before creating JSON
    std::string sanitizedPrompt = SanitizeUTF8(prompt);

    // Parse the optional comma-separated stop sequences once; both wire
    // formats accept a JSON array of strings.
    std::vector<std::string> stopSeqs;
    if (!g_OllamaStop.empty()) {
        std::stringstream ss(g_OllamaStop);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // trim whitespace
            size_t start = item.find_first_not_of(" \t");
            size_t end = item.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos)
                stopSeqs.push_back(item.substr(start, end - start + 1));
        }
    }

    nlohmann::json requestData;

    if (useOpenAi)
    {
        // OpenAI chat-completions format: a list of role/content messages.
        nlohmann::json messages = nlohmann::json::array();
        if (!g_OllamaSystemPrompt.empty())
        {
            messages.push_back({
                {"role",    "system"},
                {"content", SanitizeUTF8(g_OllamaSystemPrompt)}
            });
        }
        messages.push_back({
            {"role",    "user"},
            {"content", sanitizedPrompt}
        });

        requestData = {
            {"model",    model},
            {"messages", messages},
            {"stream",   false}
        };

        // Only include sampling parameters when the user changed them from
        // the Ollama defaults, mirroring the native path's behavior.
        if (g_OllamaNumPredict > 0)
            requestData["max_tokens"] = g_OllamaNumPredict;
        if (g_OllamaTemperature != 0.8f)
            requestData["temperature"] = g_OllamaTemperature;
        if (g_OllamaTopP != 0.95f)
            requestData["top_p"] = g_OllamaTopP;
        if (!stopSeqs.empty())
            requestData["stop"] = stopSeqs;
        if (!g_OllamaSeed.empty()) {
            try {
                requestData["seed"] = std::stoi(g_OllamaSeed);
            } catch (const std::exception&) {
                if(g_DebugEnabled) {
                    LOG_INFO("server.loading", "[Ollama Chat] Invalid seed value: {}", g_OllamaSeed);
                }
            }
        }
        // NOTE: RepeatPenalty, NumCtx, NumThreads and Think Mode are
        // Ollama-specific options that have no standard equivalent in the
        // OpenAI API, so they are not sent in this mode.
    }
    else
    {
        requestData = {
            {"model",  model},
            {"prompt", sanitizedPrompt},
            {"stream", false}
        };

        // Create options object for model parameters
        nlohmann::json options;
        bool hasOptions = false;

        // Only include if set (do not send defaults if user did not set them)
        if (g_OllamaNumPredict > 0) {
            options["num_predict"] = g_OllamaNumPredict;
            hasOptions = true;
        }
        if (g_OllamaTemperature != 0.8f) {
            options["temperature"] = g_OllamaTemperature;
            hasOptions = true;
        }
        if (g_OllamaTopP != 0.95f) {
            options["top_p"] = g_OllamaTopP;
            hasOptions = true;
        }
        if (g_OllamaRepeatPenalty != 1.1f) {
            options["repeat_penalty"] = g_OllamaRepeatPenalty;
            hasOptions = true;
        }
        if (g_OllamaNumCtx > 0) {
            options["num_ctx"] = g_OllamaNumCtx;
            hasOptions = true;
        }
        if (g_OllamaNumThreads > 0) {
            options["num_thread"] = g_OllamaNumThreads;
            hasOptions = true;
        }
        if (!g_OllamaSeed.empty()) {
            try {
                int seedValue = std::stoi(g_OllamaSeed);
                options["seed"] = seedValue;
                hasOptions = true;
            } catch (const std::exception& e) {
                if(g_DebugEnabled) {
                    LOG_INFO("server.loading", "[Ollama Chat] Invalid seed value: {}", g_OllamaSeed);
                }
            }
        }

        // Add options object if any options were set
        if (hasOptions) {
            requestData["options"] = options;
        }

        // Root-level parameters (these stay at root level)
        if (!stopSeqs.empty())
            requestData["stop"] = stopSeqs;
        if (!g_OllamaSystemPrompt.empty())
        {
            // Sanitize system prompt as well
            requestData["system"] = SanitizeUTF8(g_OllamaSystemPrompt);
        }

        if (g_ThinkModeEnableForModule)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] LLM set to Think mode.");
            }
            requestData["think"] = true;
            requestData["hidethinking"] = true;
        }
    }

    std::string requestDataStr = requestData.dump();

    // Make HTTP POST request using our custom client
    std::string responseBuffer = httpClient.Post(url, requestDataStr);

    if (responseBuffer.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: Failed to reach Ollama API at {}. Check URL configuration and network connectivity.", url);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Empty response buffer from HTTP client. Model: {}", model);
        }
        return "";
    }

    std::ostringstream extractedResponse;
    bool hitTokenLimit = false;

    try
    {
        if (useOpenAi)
        {
            // OpenAI returns a single JSON object with the text at
            // choices[0].message.content and the stop cause at
            // choices[0].finish_reason.
            nlohmann::json jsonResponse = nlohmann::json::parse(responseBuffer);

            // Surface API-level errors (LM Studio reports them this way).
            if (jsonResponse.contains("error"))
            {
                std::string errMsg = jsonResponse["error"].is_string()
                    ? jsonResponse["error"].get<std::string>()
                    : jsonResponse["error"].dump();
                LOG_ERROR("server.loading", "[OllamaChat] ERROR: API returned an error: {}", errMsg);
                return "";
            }

            if (jsonResponse.contains("choices") && jsonResponse["choices"].is_array()
                && !jsonResponse["choices"].empty())
            {
                const auto& choice = jsonResponse["choices"][0];
                if (choice.contains("message") && choice["message"].contains("content")
                    && choice["message"]["content"].is_string())
                {
                    extractedResponse << choice["message"]["content"].get<std::string>();
                }

                if (choice.contains("finish_reason") && choice["finish_reason"].is_string()
                    && choice["finish_reason"].get<std::string>() == "length")
                {
                    hitTokenLimit = true;
                }
            }
        }
        else
        {
            // Ollama streams one JSON object per line (even with stream=false
            // some versions emit a trailing newline), so parse line by line.
            std::stringstream ss(responseBuffer);
            std::string line;
            while (std::getline(ss, line))
            {
                if (line.empty() || std::all_of(line.begin(), line.end(), isspace))
                    continue;

                nlohmann::json jsonResponse = nlohmann::json::parse(line);

                if (jsonResponse.contains("response") && !jsonResponse["response"].get<std::string>().empty())
                {
                    extractedResponse << jsonResponse["response"].get<std::string>();
                }

                if (jsonResponse.contains("done_reason") && jsonResponse["done_reason"].is_string()
                    && jsonResponse["done_reason"].get<std::string>() == "length")
                {
                    hitTokenLimit = true;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: JSON parsing failed. Exception: {}", e.what());
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Response buffer content: {}", responseBuffer);
        }
        return "";
    }

    std::string botReply = extractedResponse.str();

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[OllamaChat] Debug: Extracted response field (length {}): {}", botReply.length(), botReply);
    }

    // Strip <think>...</think> blocks that thinking models (e.g. qwen3)
    // embed in the response field when think mode is not enabled via the
    // API parameter.
    {
        std::string stripped;
        size_t pos = 0;
        bool hadThinkTags = false;
        bool unclosed = false;
        while (pos < botReply.size())
        {
            size_t thinkStart = botReply.find("<think>", pos);
            if (thinkStart == std::string::npos)
            {
                stripped.append(botReply, pos, std::string::npos);
                break;
            }
            hadThinkTags = true;
            stripped.append(botReply, pos, thinkStart - pos);
            size_t thinkEnd = botReply.find("</think>", thinkStart);
            if (thinkEnd == std::string::npos)
            {
                unclosed = true;
                break;
            }
            pos = thinkEnd + 8; // length of "</think>"
        }

        if (unclosed)
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: Unclosed <think> tag detected. The model's output was likely truncated.");
            LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.ThinkModeEnableForModule = 1' in mod_ollama_chat.conf");
            LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumPredict = 0' (unlimited tokens) in mod_ollama_chat.conf");
            LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumCtx = 0' (model default context) in mod_ollama_chat.conf");
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[OllamaChat] Debug: Partial response with think tags: {}", botReply);
            }
            return "";
        }

        if (hadThinkTags)
        {
            botReply = stripped;
            // Trim whitespace left after stripping think blocks
            size_t start = botReply.find_first_not_of(" \t\n\r");
            size_t end = botReply.find_last_not_of(" \t\n\r");
            if (start == std::string::npos)
                botReply.clear();
            else
                botReply = botReply.substr(start, end - start + 1);
        }
    }

    botReply = ExtractTextBetweenDoubleQuotes(botReply);

    if (botReply.empty())
    {
        if (hitTokenLimit)
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: Model used all tokens on thinking and produced no response (done_reason: length).");
            LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Increase 'OllamaChat.NumPredict' in mod_ollama_chat.conf (current: {}) or set to 0 for unlimited.", g_OllamaNumPredict);
        }
        else
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: Empty response extracted from API. Model may not have generated any output.");
        }
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Response buffer (length {}): {}", responseBuffer.length(), responseBuffer);
        }
        return "";
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Parsed bot response: {}", botReply);

        if (g_ThinkModeEnableForModule)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Bot used think.");
            }
        }
    }

    return botReply;
}

// Helper function to check if a response is valid (not empty and not an error)
bool IsValidAPIResponse(const std::string& response)
{
    if (response.empty())
    {
        return false;
    }
    // Response is valid if it's not empty
    return true;
}

QueryManager g_queryManager;

// Interface function to submit a query.
std::future<std::string> SubmitQuery(const std::string& prompt)
{
    return g_queryManager.submitQuery(prompt);
}