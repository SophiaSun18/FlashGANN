#pragma once

#include "include/utils.cuh"
#include "include/hash_table.cuh"
#include "include/beam_management.cuh"

/** @brief Threads per PathW block: one warp per block, one block per query, matching the upstream PathWeaver. */
static constexpr unsigned PATHW_BLOCK_SIZE = 32;
static_assert(PATHW_BLOCK_SIZE == 32, "PathW and its beam primitives require one warp per block");

/*-------------------------------------------- graph --------------------------------------------*/

/** @brief Expanders per iteration. */
static constexpr unsigned SEARCH_WIDTH = 1;
/** @brief Lanes that compute one exact distance together. */
static constexpr unsigned TEAM_SIZE = 8;
/** @brief Beam ID bit that pick_expanders sets on an expanded node. */
static constexpr uint32_t INDEX_MSB_1_MASK = 0x80000000u;
/** @brief Beam IDs that pathw_inbeam compares per step, read as uint4 shared loads. */
static constexpr unsigned SCAN_CHUNK = 16;

/**
 * @brief Device copy of a fixed-degree adjacency list for the PathW kernel.
 *
 * Kernels take it by value, so it has no destructor; the owner calls release() once.
 */
class GraphGPU {
protected:
  vidType nv;           // number of vertices
  vidType maxDeg;       // maximum degree
  vidType *d_edges;     // device adjacency, nv * maxDeg IDs
public:
  /**
   * @brief Allocate d_edges and copy the host adjacency to it.
   * @param n number of vertices
   * @param d maximum degree
   * @param h_edges host adjacency, n * d IDs
   */
  GraphGPU(int n, int d, vidType *h_edges) : nv(n), maxDeg(d) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, static_cast<size_t>(n) * d * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, static_cast<size_t>(n) * d * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  /** @brief Free d_edges; safe to call twice. */
  void release() {
    if (d_edges != nullptr) {
      CUDA_SAFE_CALL(cudaFree(d_edges));
      d_edges = nullptr;
    }
  }
  /** @brief Return the vertex count. */
  inline __device__ __host__ vidType V() { return nv; }
  /** @brief Return the maximum degree. */
  inline __device__ __host__ vidType get_max_degree() { return maxDeg; }
  /**
   * @brief Return the device neighbor list of a vertex.
   * @param vid vertex ID
   * @return pointer to maxDeg neighbor IDs
   */
  inline __device__ __host__ vidType* N(vidType vid) { return d_edges + static_cast<size_t>(vid) * maxDeg; }
};

/**
 * @brief Dynamic shared memory bytes of one PathWBeamSearch block, computed by gpu_search_pathw.cu.
 *
 * Covers, in kernel order: query padded to a multiple of 4 floats so the beam stays 16-byte
 * aligned, beam plus candidate IDs, their distances, visited hash, parent list, query sign words
 * and the expander count.
 *
 * @param dim vector dimension
 * @param beam beam size
 * @param degree graph degree
 * @param bits visited hash bit length
 * @return byte count
 */
static __host__ inline size_t pathw_shared(unsigned dim, unsigned beam, unsigned degree, unsigned bits) {
    const size_t slots = effective_sort_beam_size(beam) + degree;
    return ((dim + 3) & ~3u) * sizeof(float) + slots * (sizeof(INDEX_T) + sizeof(float)) +
           hashtable_getsize(bits) * sizeof(INDEX_T) + ((dim + 31) / 32) * sizeof(uint32_t) + 2 * sizeof(uint32_t);
}

/*-------------------------------------------- distance --------------------------------------------*/
/** @brief Largest dimension that pathw_l2_distance preloads into registers; larger ones stream. */
static constexpr unsigned PATHW_L2_PRELOAD_DIM = 128;

/**
 * @brief Squared L2 distance from the query to one data row, computed by a team of lanes.
 *
 * Ported from beam_search_collab. Lane threadIdx.x % TEAM_SIZE reads 4-float chunks strided by
 * the team, preloaded into registers when dim is a multiple of 4 up to PATHW_L2_PRELOAD_DIM; an
 * xor-shuffle reduces within the team. All 32 lanes of the warp must call it.
 *
 * @tparam TEAM_SIZE lanes per distance
 * @param dim vector dimension
 * @param d_dataset_ptr row-major device data
 * @param query_ptr query vector
 * @param child_id data row to compare
 * @param valid_child false makes the lane contribute 0
 * @return the team's distance, in every lane of the team
 */
template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T pathw_l2_distance(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child) {
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);
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

/**
 * @brief Team-cooperative exact distance: pathw_l2_distance, or negated inner product when use_ip.
 * @tparam TEAM_SIZE lanes per distance
 * @param dim vector dimension
 * @param d_dataset_ptr row-major device data
 * @param query_ptr query vector
 * @param child_id data row to compare
 * @param valid_child false makes the lane contribute 0
 * @param use_ip true for negated inner product, false for squared L2
 * @return the team's distance, in every lane of the team
 */
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

/*-------------------------------------------- candidates --------------------------------------------*/
/**
 * @brief Whether child is among the first count beam IDs, ignoring the expanded flag.
 *
 * Compares SCAN_CHUNK IDs per step from uint4 loads and stops after the first step with a match;
 * the count % SCAN_CHUNK tail is scalar. The runtime count keeps beam size out of the template.
 *
 * @param child candidate ID
 * @param beam beam IDs, 16-byte aligned, expanded flag in the high bit
 * @param count beam IDs to check
 * @return true when child is in the beam
 */
static __device__ __forceinline__ bool pathw_inbeam(INDEX_T child, const INDEX_T* __restrict__ beam, uint32_t count) {
    // [1] SCAN_CHUNK IDs per step
    const uint4* chunks = reinterpret_cast<const uint4*>(beam);
    const uint32_t steps = count / SCAN_CHUNK;
    for (uint32_t step = 0; step < steps; ++step) {
        bool hit = false;
        #pragma unroll
        for (uint32_t v = 0; v < SCAN_CHUNK / 4; ++v) {
            const uint4 ids = chunks[step * (SCAN_CHUNK / 4) + v];
            hit |= (ids.x & ~INDEX_MSB_1_MASK) == child;
            hit |= (ids.y & ~INDEX_MSB_1_MASK) == child;
            hit |= (ids.z & ~INDEX_MSB_1_MASK) == child;
            hit |= (ids.w & ~INDEX_MSB_1_MASK) == child;
        }
        if (hit) return true;
    }
    // [2] scalar tail
    for (uint32_t slot = steps * SCAN_CHUNK; slot < count; ++slot) {
        if ((beam[slot] & ~INDEX_MSB_1_MASK) == child) return true;
    }
    return false;
}

/**
 * @brief Warp sort of a candidate buffer into descending score order.
 *
 * Lane l holds entries l + 32 * i in registers; after warp_sort it writes them back reversed.
 * Asserts when CANDIDATE_BUFFER_SIZE is 0 or above N_1 * 32.
 *
 * @tparam N_1 entries per lane
 * @param candidate_indices candidate IDs, permuted with the scores
 * @param candidate_distances candidate scores
 * @param CANDIDATE_BUFFER_SIZE number of entries
 */
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

    // [1] candidates to registers, padded with FLT_MAX
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
    // [2] ascending warp sort
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    __syncwarp();
    // [3] registers back to the buffer in reverse order
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

/**
 * @brief Sort a candidate buffer by descending score with the pathw_inverse width that fits capacity.
 * @param ids candidate IDs
 * @param scores candidate scores
 * @param capacity number of entries, at most 256
 */
static __device__ inline void pathw_sort(INDEX_T* ids, float* scores, unsigned capacity) {
    if (capacity <= 64) pathw_inverse<2>(ids, scores, capacity);
    else if (capacity <= 128) pathw_inverse<4>(ids, scores, capacity);
    else pathw_inverse<8>(ids, scores, capacity);
}

/**
 * @brief Pack the signs of query minus parent into 32-bit words, first dimension in the high bit.
 *
 * Only warp 0 works. All lanes compare one word of dimensions together; lane 0 stores the
 * ballot with its bits reversed. Dimensions past dim contribute zero bits.
 *
 * @tparam T vector element type
 * @param dim vector dimension
 * @param query query vector
 * @param parent parent node vector
 * @param query_sign_bits packed_dim output words
 * @param packed_dim number of 32-bit words
 */
template <typename T>
__device__ inline void pathw_build_query_sign_bits(int dim, const T* __restrict__ query, const T* __restrict__ parent,
                                                   uint32_t* __restrict__ query_sign_bits, uint32_t packed_dim) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id != 0) return;

    for (uint32_t word = 0; word < packed_dim; ++word) {
        const uint32_t d = word * WARP_SIZE + lane_id;
        const bool positive = d < static_cast<uint32_t>(dim) && query[d] > parent[d];
        const uint32_t bits = __brev(__ballot_sync(0xffffffff, positive));
        if (lane_id == 0) query_sign_bits[word] = bits;
    }
}

/**
 * @brief Fill the candidate buffer from the expanders' neighbors, keeping only the best sign-bit matches for exact distance.
 *
 * Thread i fills neighbor slot i of the single parent; a TEAM_WIDTH-lane team computes each
 * exact distance. It skips children already in the beam and does not use the visited hash. Pruned
 * slots keep their IDs with distance FLT_MAX.
 *
 * @tparam T vector element type
 * @tparam TEAM_WIDTH lanes per exact distance
 * @param dim vector dimension
 * @param parent_list beam slots of the expanders
 * @param internal_topk_list beam IDs, expanded flag in the high bit
 * @param candidate_indices candidate ID buffer
 * @param candidate_distances candidate distance buffer
 * @param query query vector
 * @param data row-major device data
 * @param sign_bits per-node neighbor sign words, sign_bit_vector_size per node
 * @param query_sign_bits scratch of packed_dim words per expander
 * @param gg device graph
 * @param graph_degree neighbors per node
 * @param candidate_buffer_size candidate buffer length
 * @param sign_bit_vector_size sign words per node, graph_degree * packed_dim
 * @param packed_dim sign words per vector
 * @param prune_ratio fraction of num_expanders * graph_degree candidates to keep
 * @param runtime_internal_topk beam slots checked for duplicates
 * @param use_ip true for negated inner product, false for squared L2
 */
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
    uint32_t graph_degree,
    uint32_t candidate_buffer_size,
    uint32_t sign_bit_vector_size,
    uint32_t packed_dim,
    float prune_ratio,
    uint32_t runtime_internal_topk,
    bool use_ip) {
    // [1] query sign words against the single expander
    const INDEX_T current_node = internal_topk_list[parent_list[0]] & ~INDEX_MSB_1_MASK;
    pathw_build_query_sign_bits<T>(dim, query, data + static_cast<size_t>(current_node) * dim,
                                    query_sign_bits, packed_dim);
    __syncthreads();

    // [2] gather neighbors of the single parent
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        INDEX_T child_id = gg.N(current_node)[i];
        if (pathw_inbeam(child_id, internal_topk_list, runtime_internal_topk)) child_id = MAX_INDEX;
        bool valid_child = child_id != MAX_INDEX && child_id < gg.V();

        DISTANCE_T score = -FLT_MAX;
        if (valid_child) {
            int direction = 0;
            const uint32_t* parent_query_sign = query_sign_bits;
            const uint32_t* neighbor_sign = sign_bits + static_cast<size_t>(current_node) * sign_bit_vector_size + static_cast<size_t>(i) * packed_dim;
            for (uint32_t word = 0; word < packed_dim; ++word) {
                direction -= __popc(parent_query_sign[word] ^ neighbor_sign[word]);
            }
            score = static_cast<DISTANCE_T>(direction);
        }

        candidate_indices[i] = child_id;
        candidate_distances[i] = score;
    }
    __syncthreads();

    // [3] sort by descending score
    pathw_sort(
        candidate_indices, candidate_distances, candidate_buffer_size);
    __syncthreads();

    // [4] keep count from prune_ratio, clamped to [1, candidate_buffer_size]
    int keep_count = static_cast<int>(static_cast<float>(graph_degree) * prune_ratio);
    if (keep_count < 1) keep_count = 1;
    if (keep_count > static_cast<int>(candidate_buffer_size)) {
        keep_count = static_cast<int>(candidate_buffer_size);
    }

    // [5] as upstream, set pruned distances to FLT_MAX and keep their IDs
    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        if (i >= static_cast<uint32_t>(keep_count)) {
            candidate_distances[i] = FLT_MAX;

        }
    }
    __syncthreads();

    // [6] exact distances of the kept candidates, work padded to whole warps for the shuffles
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

/**
 * @brief Fill the candidate buffer from the expanders' neighbors not yet in the visited hash, with exact distances.
 *
 * Thread i fills neighbor slot i of the single parent and inserts its child into the hash;
 * a TEAM_WIDTH-lane team computes each exact distance. Rejected slots get MAX_INDEX and FLT_MAX.
 *
 * @tparam T vector element type
 * @tparam TEAM_WIDTH lanes per exact distance
 * @param dim vector dimension
 * @param parent_list beam slots of the expanders
 * @param internal_topk_list beam IDs, expanded flag in the high bit
 * @param candidate_indices candidate ID buffer
 * @param candidate_distances candidate distance buffer
 * @param query query vector
 * @param data row-major device data
 * @param visited_hash visited hash table
 * @param bitlen visited hash bit length
 * @param gg device graph
 * @param graph_degree neighbors per node
 * @param candidate_buffer_size candidate buffer length
 * @param use_ip true for negated inner product, false for squared L2
 */
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
    uint32_t graph_degree,
    uint32_t candidate_buffer_size,
    bool use_ip) {
    // [1] gather real neighbors of the single parent and drop the visited ones
    const INDEX_T current_node = internal_topk_list[parent_list[0]] & ~INDEX_MSB_1_MASK;
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        INDEX_T child_id = gg.N(current_node)[i];
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

    // [2] exact distances of the remaining candidates, work padded to whole warps for the shuffles
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
