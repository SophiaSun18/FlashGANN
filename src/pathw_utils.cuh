#pragma once

#include "include/utils.cuh"

static constexpr unsigned PATHW_BLOCK_SIZE = 32;
static_assert(PATHW_BLOCK_SIZE == 32, "PathW and its beam primitives require one warp per block");

static constexpr unsigned SEARCH_WIDTH = 1;
static constexpr unsigned TEAM_SIZE = 8;
static constexpr uint32_t INDEX_MSB_1_MASK = 0x80000000u;

class GraphGPU {
protected:
  vidType nv;           // number of vertices
  vidType maxDeg;       // maximun degree
  vidType *d_edges;
public:
  GraphGPU(int n, int d, vidType *h_edges) : nv(n), maxDeg(d) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, static_cast<size_t>(n) * d * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, static_cast<size_t>(n) * d * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  void release() {
    if (d_edges != nullptr) {
      CUDA_SAFE_CALL(cudaFree(d_edges));
      d_edges = nullptr;
    }
  }
  inline __device__ __host__ vidType V() { return nv; }
  inline __device__ __host__ vidType get_max_degree() { return maxDeg; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_edges + static_cast<size_t>(vid) * maxDeg; }
};

/** @brief Shared layout: query, beam/candidate IDs and distances, hash, query signs. */
static __host__ inline size_t pathw_shared(unsigned dim, unsigned beam, unsigned degree, unsigned bits) {
    const size_t slots = effective_sort_beam_size(beam) + round_up_power2_u32(degree);
    return dim * sizeof(float) + slots * (sizeof(INDEX_T) + sizeof(float)) +
           hashtable_getsize(bits) * sizeof(INDEX_T) + ((dim + 31) / 32) * sizeof(uint32_t) + 2 * sizeof(uint32_t);
}

// L2 preload threshold; larger dimensions use the streaming loop below.
static constexpr unsigned PATHW_L2_PRELOAD_DIM = 128;

/** @brief Original beam_search_collab eight-lane exact-distance helpers. */
template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T pathw_l2_distance(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child) {
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T norm2 = 0;
    if (valid_child) {
        const DATA_T* child_data_ptr = d_dataset_ptr + static_cast<size_t>(child_id) * dim;
        const uint32_t vec_dim = dim & ~(vlen - 1);
        if (dim <= PATHW_L2_PRELOAD_DIM && (dim & (vlen - 1)) == 0) {
            constexpr unsigned reg_nelem = (PATHW_L2_PRELOAD_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
            float4 dl_buff[reg_nelem];
            const float4* child_vec_ptr = reinterpret_cast<const float4*>(child_data_ptr);
            #pragma unroll
            for (uint32_t e = 0; e < reg_nelem; e++) {
                const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
                if (k >= dim) break;
                dl_buff[e] = child_vec_ptr[k / vlen];
            }
            #pragma unroll
            for (uint32_t e = 0; e < reg_nelem; e++) {
                const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
                if (k >= dim) break;
                DISTANCE_T d = query_ptr[k];
                norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
                d = query_ptr[k + 1];
                norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
                d = query_ptr[k + 2];
                norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
                d = query_ptr[k + 3];
                norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);
            }
        } else if ((dim & (vlen - 1)) == 0) {
            const float4* child_vec_ptr = reinterpret_cast<const float4*>(child_data_ptr);
            for (uint32_t k = lane_id * vlen; k < vec_dim; k += TEAM_SIZE * vlen) {
                const float4 dl = child_vec_ptr[k / vlen];
                DISTANCE_T d = query_ptr[k];
                norm2 += (d - dl.x) * (d - dl.x);
                d = query_ptr[k + 1];
                norm2 += (d - dl.y) * (d - dl.y);
                d = query_ptr[k + 2];
                norm2 += (d - dl.z) * (d - dl.z);
                d = query_ptr[k + 3];
                norm2 += (d - dl.w) * (d - dl.w);
            }
        } else {
            for (uint32_t k = lane_id * vlen; k < vec_dim; k += TEAM_SIZE * vlen) {
                DISTANCE_T d = query_ptr[k];
                DISTANCE_T x = child_data_ptr[k];
                norm2 += (d - x) * (d - x);
                d = query_ptr[k + 1];
                x = child_data_ptr[k + 1];
                norm2 += (d - x) * (d - x);
                d = query_ptr[k + 2];
                x = child_data_ptr[k + 2];
                norm2 += (d - x) * (d - x);
                d = query_ptr[k + 3];
                x = child_data_ptr[k + 3];
                norm2 += (d - x) * (d - x);
            }
            for (uint32_t k = vec_dim + lane_id; k < dim; k += TEAM_SIZE) {
                const DISTANCE_T d = query_ptr[k] - child_data_ptr[k];
                norm2 += d * d;
            }
        }
    }
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T pathw_distance(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child,
    bool use_ip) {
    if (!use_ip) {
        return pathw_l2_distance<TEAM_SIZE>(
            dim, d_dataset_ptr, query_ptr, child_id, valid_child);
    }
    unsigned lane_id = threadIdx.x % TEAM_SIZE;
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T dot = 0;
    if (valid_child) {
        const DATA_T* child_data_ptr = d_dataset_ptr + static_cast<size_t>(child_id) * dim;
        for (uint32_t k = lane_id; k < dim; k += TEAM_SIZE) {
            dot += query_ptr[k] * child_data_ptr[k];
        }
    }
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(full_mask, dot, offset);
    }
    return -dot;
}

template <unsigned N_1>
__device__ void pathw_inverse
(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    const unsigned lane_id = threadIdx.x % 32;
    if (CANDIDATE_BUFFER_SIZE == 0 || CANDIDATE_BUFFER_SIZE > N_1 * 32) 
    {
        printf("CANDIDATE_BUFFER_SIZE exceeds this warp sort capacity\n");
        assert(false);
    }
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = FLT_MAX;
            val_1[i] = MAX_INDEX;
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    __syncwarp();
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

static __device__ inline void pathw_sort(INDEX_T* ids, float* scores, unsigned capacity) {
    if (capacity <= 64) pathw_inverse<2>(ids, scores, capacity);
    else if (capacity <= 128) pathw_inverse<4>(ids, scores, capacity);
    else pathw_inverse<8>(ids, scores, capacity);
}

template <typename T>
__device__ inline void pathw_build_query_sign_bits(int dim, const T* __restrict__ query, const T* __restrict__ parent,
                                                   uint32_t* __restrict__ query_sign_bits, uint32_t packed_dim) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id != 0) return;

    for (uint32_t word = lane_id; word < packed_dim; word += WARP_SIZE) {
        uint32_t bits = 0;
        const uint32_t base_dim = word * 32;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            const uint32_t d = base_dim + bit;
            if (d < static_cast<uint32_t>(dim) && query[d] > parent[d]) {
                bits |= (1u << (31 - bit));
            }
        }
        query_sign_bits[word] = bits;
    }
}

template <typename T, uint32_t TEAM_WIDTH>
__device__ __forceinline__ void pathw_compute_candidates_with_signbit_pruning(
    int dim,
    const INDEX_T* __restrict__ parent_list,
    const INDEX_T* __restrict__ internal_topk_list,
    INDEX_T* __restrict__ candidate_indices,
    DISTANCE_T* __restrict__ candidate_distances,
    const T* __restrict__ query,
    const T* __restrict__ data,
    const uint32_t* __restrict__ sign_bits,
    uint32_t* __restrict__ query_sign_bits,
    GraphGPU gg,
    uint32_t num_expanders,
    uint32_t graph_degree,
    uint32_t candidate_work_count,
    uint32_t candidate_buffer_size,
    uint32_t sign_bit_vector_size,
    uint32_t packed_dim,
    float prune_ratio,
    uint32_t runtime_internal_topk,
    bool use_ip) {
    for (uint32_t parent_idx = 0; parent_idx < num_expanders; ++parent_idx) {
        const INDEX_T parent_slot = parent_list[parent_idx];
        const INDEX_T current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
        pathw_build_query_sign_bits<T>(
            dim, query, data + static_cast<size_t>(current_node) * dim,
            query_sign_bits + static_cast<size_t>(parent_idx) * packed_dim, packed_dim);
        __syncthreads();
    }

    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        const bool in_work_range = i < candidate_work_count;
        const uint32_t parent_idx = in_work_range ? i / graph_degree : 0;
        const uint32_t neighbor_idx = in_work_range ? i - parent_idx * graph_degree : 0;
        const bool active_parent = in_work_range && parent_idx < num_expanders;
        INDEX_T current_node = MAX_INDEX;
        INDEX_T child_id = MAX_INDEX;
        if (active_parent) {
            const INDEX_T parent_slot = parent_list[parent_idx];
            current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
            child_id = gg.N(current_node)[neighbor_idx];
        }
        bool valid_child = child_id != MAX_INDEX && child_id < gg.V();
        for (unsigned slot = 0; valid_child && slot < runtime_internal_topk; ++slot)
            valid_child = child_id != (internal_topk_list[slot] & ~INDEX_MSB_1_MASK);
        if (!valid_child) child_id = MAX_INDEX;

        DISTANCE_T score = -FLT_MAX;
        // load the precomputed sign bits of the child, and compute the matching score
        if (valid_child) {
            int direction = 0;
            const uint32_t* parent_query_sign = query_sign_bits + static_cast<size_t>(parent_idx) * packed_dim;
            const uint32_t* neighbor_sign = sign_bits + static_cast<size_t>(current_node) * sign_bit_vector_size + static_cast<size_t>(neighbor_idx) * packed_dim;
            for (uint32_t word = 0; word < packed_dim; ++word) {
                direction -= __popc(parent_query_sign[word] ^ neighbor_sign[word]);
            }
            score = static_cast<DISTANCE_T>(direction);
        }

        candidate_indices[i] = child_id;
        candidate_distances[i] = score;
    }
    __syncthreads();

    pathw_sort(
        candidate_indices, candidate_distances, candidate_buffer_size);
    __syncthreads();

    const uint32_t active_candidate_count = num_expanders * graph_degree;
    int keep_count = static_cast<int>(static_cast<float>(active_candidate_count) * prune_ratio);
    if (keep_count < 1) keep_count = 1;
    if (keep_count > static_cast<int>(candidate_buffer_size)) {
        keep_count = static_cast<int>(candidate_buffer_size);
    }

    // Match upstream: prune distances while retaining candidate IDs.
    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        if (i >= static_cast<uint32_t>(keep_count)) {
            candidate_distances[i] = FLT_MAX;

        }
    }
    __syncthreads();

    // compute the exact distances for the kept candidates
    const uint32_t distance_work_count = static_cast<uint32_t>(keep_count) * TEAM_WIDTH;
    const uint32_t padded_work_count = ((distance_work_count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for (uint32_t work = threadIdx.x; work < padded_work_count; work += blockDim.x) {
        const bool in_range = work < distance_work_count;
        const uint32_t i = in_range ? work / TEAM_WIDTH : 0;
        const INDEX_T child_id = in_range ? candidate_indices[i] : MAX_INDEX;
        const bool valid_child = in_range && child_id != MAX_INDEX;
        DISTANCE_T dist = pathw_distance<TEAM_WIDTH>(
            dim, data, query, valid_child ? child_id : 0, valid_child, use_ip);
        if (in_range && (work % TEAM_WIDTH) == 0) {
            candidate_distances[i] = valid_child ? dist : FLT_MAX;
        }
    }
}

template <typename T, uint32_t TEAM_WIDTH>
__device__ __forceinline__ void pathw_compute_candidates_plain(
    int dim,
    const INDEX_T* __restrict__ parent_list,
    const INDEX_T* __restrict__ internal_topk_list,
    INDEX_T* __restrict__ candidate_indices,
    DISTANCE_T* __restrict__ candidate_distances,
    const T* __restrict__ query,
    const T* __restrict__ data,
    INDEX_T* __restrict__ visited_hash,
    int bitlen,
    GraphGPU gg,
    uint32_t num_expanders,
    uint32_t graph_degree,
    uint32_t candidate_work_count,
    uint32_t candidate_buffer_size,
    bool use_ip) {
    // initialize the candidate buffer and filter out the visited ones
    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        const bool in_work_range = i < candidate_work_count;
        const uint32_t parent_idx = in_work_range ? i / graph_degree : 0;
        const uint32_t neighbor_idx = in_work_range ? i - parent_idx * graph_degree : 0;
        INDEX_T child_id = MAX_INDEX;
        if (in_work_range && parent_idx < num_expanders) {
            const INDEX_T parent_slot = parent_list[parent_idx];
            const INDEX_T current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
            child_id = gg.N(current_node)[neighbor_idx];
        }
        if (child_id != MAX_INDEX && child_id < gg.V()) {
            if (hashtable_insert(visited_hash, bitlen, child_id) == 0) {
                child_id = MAX_INDEX;
                candidate_distances[i] = FLT_MAX;
            } else {
                candidate_distances[i] = 0;
            }
        } else {
            child_id = MAX_INDEX;
            candidate_distances[i] = FLT_MAX;
        }
        candidate_indices[i] = child_id;
    }
    __syncthreads();

    // compute the exact distances for the candidates that are not filtered out
    const uint32_t distance_work_count = candidate_buffer_size * TEAM_WIDTH;
    const uint32_t padded_work_count = ((distance_work_count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for (uint32_t work = threadIdx.x; work < padded_work_count; work += blockDim.x) {
        const bool in_range = work < distance_work_count;
        const uint32_t i = in_range ? work / TEAM_WIDTH : 0;
        const INDEX_T child_id = in_range ? candidate_indices[i] : MAX_INDEX;
        const bool valid_child = in_range && child_id != MAX_INDEX;
        DISTANCE_T dist = pathw_distance<TEAM_WIDTH>(
            dim, data, query, valid_child ? child_id : 0, valid_child, use_ip);
        if (in_range && (work % TEAM_WIDTH) == 0) {
            candidate_distances[i] = valid_child ? dist : FLT_MAX;
        }
    }
}
