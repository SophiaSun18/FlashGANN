#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

enum MetricType { METRIC_L2, METRIC_IP };

inline MetricType g_metric_type = METRIC_L2;

inline constexpr std::array<std::string_view, 1> kIpMetricDatasets = {
    "text2image1m",
};

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

inline MetricType infer_metric_from_dataset_path(const char* path) {
    return path ? infer_metric_from_dataset_path(std::string(path)) : METRIC_L2;
}

inline const char* metric_name(MetricType metric) {
    return metric == METRIC_IP ? "ip" : "l2";
}
