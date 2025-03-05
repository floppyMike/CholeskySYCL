#pragma once

#include "gpu.hpp"

inline auto getduration(sycl::event &e) -> uint64_t {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>();
}
