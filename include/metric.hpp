#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

/** @brief Distance metric: squared L2 or negated inner product. */
enum MetricType { METRIC_L2, METRIC_IP };

/** @brief Process-wide metric, set by main.cu from the first data file path. */
inline MetricType g_metric_type = METRIC_L2;

/** @brief Dataset name substrings whose paths select METRIC_IP. */
inline constexpr std::array<std::string_view, 1> kIpMetricDatasets = {
    "text2image1m",
};

/**
 * @brief Choose the metric from a dataset path by case-insensitive match against kIpMetricDatasets.
 * @param path dataset file path
 * @return METRIC_IP on a match, otherwise METRIC_L2
 */
inline MetricType infer_metric_from_dataset_path(const std::string& path) {
    std::string lower_path = path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    for (std::string_view dataset : kIpMetricDatasets) {
        if (lower_path.find(dataset) != std::string::npos) {
            return METRIC_IP;
        }
    }
    return METRIC_L2;
}

/**
 * @brief C-string overload of infer_metric_from_dataset_path.
 * @param path dataset file path, or nullptr
 * @return the inferred metric, METRIC_L2 for nullptr
 */
inline MetricType infer_metric_from_dataset_path(const char* path) {
    return path ? infer_metric_from_dataset_path(std::string(path)) : METRIC_L2;
}

/**
 * @brief Name a metric for log output.
 * @param metric metric to name
 * @return "ip" or "l2"
 */
inline const char* metric_name(MetricType metric) {
    return metric == METRIC_IP ? "ip" : "l2";
}
