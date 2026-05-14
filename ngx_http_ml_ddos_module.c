#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_thread_pool.h>
#include <onnxruntime_c_api.h>

#include <math.h>

#define LOG_PREFIX "ML_DDOS: "
#if NGX_DEBUG
#    define NGX_ASSERT(expr, log)                                           \
        do {                                                                \
            if (!(expr)) {                                                  \
                ngx_log_error(NGX_LOG_EMERG, log, NGX_ERROR,                \
                              LOG_PREFIX "Assertion failed at %s:%d -- %s", \
                              __FILE__, __LINE__, #expr);                   \
                ngx_debug_point();                                          \
            }                                                               \
        } while (0)
#else
#    define NGX_ASSERT(...)
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#    include <assert.h>
#    include <stdalign.h>

#    define MLDS_STATIC_ASSERT(expr, log) static_assert(expr, LOG_PREFIX log)
#else
#    define MLDS_STATIC_ASSERT_CONCAT(a, b) a##b
#    define MLDS_STATIC_ASSERT_LINE(a, b) MLDS_STATIC_ASSERT_CONCAT(a, b)
#    define MLDS_STATIC_ASSERT(expr, msg)                               \
        typedef char MLDS_STATIC_ASSERT_LINE(mlds_static_assert_line_,  \
                                             __LINE__)[(expr) ? 1 : -1]
#    define alignof __alignof__
#endif

typedef struct {
    ngx_str_t model_path;
    ngx_shm_zone_t *shm_buffer;
    const OrtApi *ort_api;
    OrtEnv *ort_env;
    OrtSessionOptions *ort_session_options;
    OrtSession *ort_session;
    OrtMemoryInfo *ort_memory_info;
} ngx_http_ml_ddos_main_conf_t;

typedef struct {
    ngx_flag_t enabled;
    ngx_flag_t sampling;
    ngx_thread_pool_t *thread_pool;
    float block_threshold;
    float limit_threshold;
} ngx_http_ml_ddos_loc_conf_t;

static void *ngx_http_ml_ddos_create_main_conf(ngx_conf_t *cf);
static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_ml_ddos_merge_loc_conf(ngx_conf_t *cf, void *parent,
                                             void *child);
static ngx_int_t ngx_http_ml_ddos_init(ngx_conf_t *cf);
static ngx_int_t ngx_http_ml_ddos_init_process(ngx_cycle_t *cycle);
static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle);

static char *ngx_http_ml_ddos_path(ngx_conf_t *cf, ngx_command_t *cmd,
                                   void *conf);
static char *ngx_http_ml_ddos_enable(ngx_conf_t *cf, ngx_command_t *cmd,
                                     void *conf);

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r);

static ngx_command_t ngx_http_ml_ddos_commands[] = {
    {ngx_string("ml_ddos_path"),          // directive
     NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1, // location context
     ngx_http_ml_ddos_path,               // configuration setup function
     0,                                   // local offset
     0,                                   // configuration offset
     NULL},

    {ngx_string("ml_ddos"),              // directive
     NGX_HTTP_LOC_CONF | NGX_CONF_1MORE, // location context
     ngx_http_ml_ddos_enable,            // configuration setup function
     NGX_HTTP_LOC_CONF_OFFSET,           // local offset
     offsetof(ngx_http_ml_ddos_loc_conf_t, enabled), // configuration offset
     NULL},

    ngx_null_command};

static ngx_http_module_t ngx_http_ml_ddos_module_ctx = {
    NULL,                              // preconfiguration
    ngx_http_ml_ddos_init,             // postconfiguration
    ngx_http_ml_ddos_create_main_conf, // create main configuration
    NULL,                              // init main configuration
    NULL,                              // create server configuration
    NULL,                              // merge server configuration
    ngx_http_ml_ddos_create_loc_conf,  // create local configuration
    ngx_http_ml_ddos_merge_loc_conf    // merge local configuration
};

ngx_module_t ngx_http_ml_ddos_module = {
    NGX_MODULE_V1,
    &ngx_http_ml_ddos_module_ctx,  // module context
    ngx_http_ml_ddos_commands,     // module directives
    NGX_HTTP_MODULE,               // module type
    NULL,                          // init master
    NULL,                          // init module
    ngx_http_ml_ddos_init_process, // init process
    NULL,                          // init thread
    NULL,                          // exit thread
    ngx_http_ml_ddos_exit_process, // exit process
    NULL,                          // exit master
    NGX_MODULE_V1_PADDING};

/// ===== CONFIG =====

static void *ngx_http_ml_ddos_create_main_conf(ngx_conf_t *cf) {
    return ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_main_conf_t));
}

static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf) {
    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_loc_conf_t));
    lcf->block_threshold = NAN;
    lcf->limit_threshold = NAN;
    return lcf;
}

static char *ngx_http_ml_ddos_merge_loc_conf(ngx_conf_t *cf, void *parent,
                                             void *child) {
    ngx_http_ml_ddos_loc_conf_t *prev = parent;
    ngx_http_ml_ddos_loc_conf_t *conf = child;

    ngx_conf_merge_value(conf->enabled, prev->enabled, 0);
    ngx_conf_merge_value(conf->sampling, prev->sampling, 0);

    if (conf->thread_pool == NULL)
        conf->thread_pool = prev->thread_pool;

    if (isnan(conf->block_threshold))
        conf->block_threshold =
            isnan(prev->block_threshold) ? 0.85f : prev->block_threshold;
    if (isnan(conf->limit_threshold))
        conf->limit_threshold =
            isnan(prev->limit_threshold) ? 0.65f : prev->limit_threshold;

    if (conf->limit_threshold >= conf->block_threshold) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                           LOG_PREFIX "limit must be less than block");
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

/// ===== PROCESS ======

/**
 * Initializes ONNX model with specific session options:
 *
 * Threading:
 * 1. Intra-op threads: 0 (because we use internal nginx thread pool)
 * 2. Inter-op threads: 0 (same reason)
 * 3. Execution mode: SEQUENTIAL (same reason)
 * 4. Spinning Behavior: 1 (use all CPU it can)
 *
 * Graph optimization level:
 * 1. Graph optimization: DISABLE_ALL (for small models it can be slower)
 *
 * Memory managment:
 * 1. Memory pattern: 1 (we use same size of input each time)
 * 2. Arena allocator: 1 (more efficient than casual allocations)
 *
 * Logging:
 * 1. Log level and profiling: depends on NGX_DEBUG
 */
static ngx_int_t ngx_http_ml_ddos_init_process(ngx_cycle_t *cycle) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_cycle_get_module_main_conf(cycle, ngx_http_ml_ddos_module);
    if (mcf == NULL || mcf->model_path.len == 0) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "model path not configured");
        return NGX_ERROR;
    }

    u_char *model_path_cstr = ngx_pnalloc(cycle->pool, mcf->model_path.len + 1);
    if (!model_path_cstr)
        return NGX_ERROR;
    ngx_memcpy(model_path_cstr, mcf->model_path.data, mcf->model_path.len);
    model_path_cstr[mcf->model_path.len] = '\0';

    ngx_file_info_t fi;
    if (ngx_file_info((const char *)model_path_cstr, &fi) == NGX_FILE_ERROR) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno,
                      LOG_PREFIX "model not found: %s", model_path_cstr);
        return NGX_ERROR;
    }

    if (!(mcf->ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION))) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "failed to get ONNX API");
        return NGX_ERROR;
    }

    OrtStatus *status;
#define ONNX_ASSERT(expr, err) \
    if ((status = (expr)))     \
        goto err;

#if NGX_DEBUG
#    define ONNX_LOG_LEVEL ORT_LOGGING_LEVEL_VERBOSE
#else
#    define ONNX_LOG_LEVEL ORT_LOGGING_LEVEL_ERROR
#endif
    ONNX_ASSERT(
        mcf->ort_api->CreateEnv(ONNX_LOG_LEVEL, "ngx_ml_ddos", &mcf->ort_env),
        error_env);

    ONNX_ASSERT(mcf->ort_api->CreateSessionOptions(&mcf->ort_session_options),
                error_options);

    ONNX_ASSERT(mcf->ort_api->SetIntraOpNumThreads(mcf->ort_session_options, 0),
                error_options);
    ONNX_ASSERT(mcf->ort_api->SetInterOpNumThreads(mcf->ort_session_options, 0),
                error_options);

    ONNX_ASSERT(mcf->ort_api->SetSessionExecutionMode(mcf->ort_session_options,
                                                      ORT_SEQUENTIAL),
                error_options);
    ONNX_ASSERT(
        mcf->ort_api->AddSessionConfigEntry(
            mcf->ort_session_options, "session.intra_op.allow_spinning", "1"),
        error_options);

    ONNX_ASSERT(mcf->ort_api->EnableMemPattern(mcf->ort_session_options),
                error_options);
    ONNX_ASSERT(mcf->ort_api->EnableCpuMemArena(mcf->ort_session_options),
                error_options);

    ONNX_ASSERT(mcf->ort_api->CreateSession(
                    mcf->ort_env, (const char *)model_path_cstr,
                    mcf->ort_session_options, &mcf->ort_session),
                error_session);

    ONNX_ASSERT(mcf->ort_api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                                  OrtMemTypeDefault,
                                                  &mcf->ort_memory_info),
                error_memory);

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, NGX_OK,
                  LOG_PREFIX "initialized with model: %V", &mcf->model_path);

    return NGX_OK;

error_memory:
    mcf->ort_api->ReleaseSession(mcf->ort_session);
error_session:
    mcf->ort_api->ReleaseSessionOptions(mcf->ort_session_options);
error_options:
    mcf->ort_api->ReleaseEnv(mcf->ort_env);
error_env:
    ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                  LOG_PREFIX "ONNX initialize failed: %s",
                  mcf->ort_api->GetErrorMessage(status));
    mcf->ort_api->ReleaseStatus(status);
    return NGX_ERROR;

#undef ONNX_ASSERT
}

static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_cycle_get_module_main_conf(cycle, ngx_http_ml_ddos_module);
    NGX_ASSERT(mcf->ort_session && mcf->ort_session_options && mcf->ort_env &&
                   mcf->ort_memory_info,
               cycle->log);

    if (mcf->ort_memory_info)
        mcf->ort_api->ReleaseMemoryInfo(mcf->ort_memory_info);
    if (mcf->ort_session)
        mcf->ort_api->ReleaseSession(mcf->ort_session);
    if (mcf->ort_session_options)
        mcf->ort_api->ReleaseSessionOptions(mcf->ort_session_options);
    if (mcf->ort_env)
        mcf->ort_api->ReleaseEnv(mcf->ort_env);

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, NGX_OK,
                  LOG_PREFIX "ONNX runtime shutdown");
}

/// ===== FEATURES ======

#define NGX_HTTP_ML_DDOS_BUFFER_NAME "ML_DDOS_SHARED_BUFFER"
#define NGX_HTTP_ML_DDOS_BUFFER_SIZE 1024
#define NGX_HTTP_ML_DDOS_TABLE_SIZE (NGX_HTTP_ML_DDOS_BUFFER_SIZE * 4)

MLDS_STATIC_ASSERT(NGX_HTTP_ML_DDOS_BUFFER_SIZE <= 65535,
                   "Buffer size must not exceed uint16_t max capacity (65535)");

#define POWER_OF_2_CHECK(n)                                   \
    MLDS_STATIC_ASSERT(((n) > 0) && (((n) & ((n) - 1))) == 0, \
                       #n " size must be a power of two!")

POWER_OF_2_CHECK(NGX_HTTP_ML_DDOS_BUFFER_SIZE);
POWER_OF_2_CHECK(NGX_HTTP_ML_DDOS_TABLE_SIZE);

#define NGX_HTTP_ML_DDOS_BUFFER_LIMIT(n, buffer) ((n) & (buffer - 1))

typedef struct ngx_http_ml_ddos_features_flags {
    uint32_t http_version : 2;
    uint32_t http_req_type : 2;
    uint32_t has_user_agent : 2;
    uint32_t has_ua_browser : 2;
    uint32_t has_accept : 2;
    uint32_t has_accept_language : 2;
    uint32_t has_accept_encoding : 2;
    uint32_t has_content_type : 2;
    uint32_t has_cookies : 2;
    uint32_t has_authorization : 2;
    uint32_t has_referer : 2;
    uint32_t has_host : 2;
    uint32_t has_upgrade : 2;
    uint32_t has_transfer_encoding : 2;
    uint32_t keepalive : 2;
    uint32_t quoted_uri : 2;
} mlds_features_flags_t;

typedef struct ngx_http_ml_ddos_features_params {
    uint32_t ns_delta;         // delta between requests
    uint32_t ip_request_count; // request count per user
    uint32_t uri_length;
    uint32_t args_length;
    uint32_t header_count;
    uint32_t content_length;
    uint32_t connection_requests;
    mlds_features_flags_t flags;
} mlds_features_params_t;

typedef union ngx_http_ml_ddos_features {
    mlds_features_params_t params;
    uint32_t data[8];
} mlds_features_t;

typedef struct ngx_http_ml_ddos_metadata {
    uint64_t timestamp;
    uint32_t uid;
    uint32_t __padding__;
} mlds_metadata_t;

MLDS_STATIC_ASSERT(sizeof(mlds_features_flags_t) == sizeof(uint32_t),
                   "mlds_features_flags_t must be packed in uint32_t");
MLDS_STATIC_ASSERT(
    sizeof(mlds_features_params_t) == sizeof(uint32_t) * 8,
    "mlds_features_params_t size must be exactly 8 * sizeof(uint32_t)");
MLDS_STATIC_ASSERT(sizeof(mlds_features_t) == sizeof(mlds_features_params_t),
                   "mlds_features_t size must match mlds_features_params_t");
MLDS_STATIC_ASSERT(sizeof(mlds_metadata_t) == sizeof(uint64_t) * 2,
                   "mlds_metadata_t size must be exactly 2 of uint64_t");

#define MLDS_CHECK_ALIGN_OF_STRUCTS(name, type)                   \
    MLDS_STATIC_ASSERT(alignof(name) == sizeof(type),             \
                       #name " align must be the size of " #type)

MLDS_CHECK_ALIGN_OF_STRUCTS(mlds_features_flags_t, uint32_t);
MLDS_CHECK_ALIGN_OF_STRUCTS(mlds_features_params_t, uint32_t);
MLDS_CHECK_ALIGN_OF_STRUCTS(mlds_features_t, uint32_t);
MLDS_CHECK_ALIGN_OF_STRUCTS(mlds_metadata_t, uint64_t);

/**
 * This buffer is located in shared memory across processes and holds data
 * about time-series features of requests as synchronized static ring
 * buffers. In addition, for O(1) user id counting in the buffer, we also
 * allocate an direct-mapped hash table with MurmurHash3 algorithm for
 * uint32_t.
 *
 * NOTE ON COLLISIONS:
 * To maintain strictly non-blocking behaviour and avoid complex chaining or
 * dynamic allocation in shared memory, we accept hash collisions in the
 * uid_table. In the event of a collision, the IP request count may reflect
 * and aggregate of multiple users. For DDoS detection, this "false
 * positive" pressure is acceptable as it leans toward a conservative
 * secrurity posture without sacrificing the throughput of the Nginx worker
 * event loop.
 *
 * Buffers are split to minimize time spent holding the shared mutex:
 * specifically, this allows for an efficient copy of the features buffer
 * into an ONNX [BUFFER_SIZE, 8] matrix.
 *
 * goes this way ->            when reached the end, tail_idx is reset
 * +----------------------+------------------------------------------+
 * | features             |                                          |
 * +----------------------+------------------------------------------+
 * | meta                 |                                          |
 * +----------------------+------------------------------------------+
 * ^                      ^                                          ^
 * buffers start       tail_idx                                BUFFER_SIZE
 */
typedef struct ngx_http_ml_ddos_buffer {
    uint32_t tail_idx;
    mlds_features_t features[NGX_HTTP_ML_DDOS_BUFFER_SIZE];
    mlds_metadata_t meta[NGX_HTTP_ML_DDOS_BUFFER_SIZE];
    uint16_t uid_table[NGX_HTTP_ML_DDOS_TABLE_SIZE];
} mlds_buffer_t;

static ngx_int_t mlds_init_shm(ngx_shm_zone_t *zone, void *data) {
    if (data) {
        zone->data = data;
        return NGX_OK;
    }

    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *)zone->shm.addr;
    mlds_buffer_t *shm = ngx_slab_calloc(shpool, sizeof(*shm));
    if (!shm)
        return NGX_ERROR;

    zone->data = shm;

    return NGX_OK;
}

static ngx_inline ngx_shm_zone_t *mlds_create_shm(ngx_conf_t *cf,
                                                  void *module) {
    static ngx_str_t shm_name = ngx_string(NGX_HTTP_ML_DDOS_BUFFER_NAME);
    ngx_shm_zone_t *shm_zone = ngx_shared_memory_add(
        cf, &shm_name, 8192 + sizeof(mlds_buffer_t), module);
    if (!shm_zone)
        return NULL;

    shm_zone->init = mlds_init_shm;

    return shm_zone;
}

/// MurmurHash3 mixing constants to map a uint32_t to [0, TABLE_SIZE - 1]
static ngx_inline uint32_t hash_uint32(uint32_t key) {
#define MM3_FINALIZER_MUL1 0x85ebca6b
#define MM3_FINALIZER_MUL2 0xc2b2ae35

    key ^= key >> 16;
    key *= MM3_FINALIZER_MUL1;
    key ^= key >> 13;
    key *= MM3_FINALIZER_MUL2;
    key ^= key >> 16;

    return NGX_HTTP_ML_DDOS_BUFFER_LIMIT(key, NGX_HTTP_ML_DDOS_TABLE_SIZE);
}

static ngx_inline uint64_t shm_buffer_last_timestamp(mlds_buffer_t *buffer) {
    return buffer->meta[buffer->tail_idx].timestamp;
}

static ngx_inline uint16_t shm_buffer_uid_count(mlds_buffer_t *buffer,
                                                uint32_t uid) {
    return buffer->uid_table[hash_uint32(uid)];
}

/// Pushes a new element into the sync ring buffers and updates the uid table
static ngx_inline void shm_buffer_push_element(mlds_buffer_t *buffer,
                                               mlds_features_t *features,
                                               mlds_metadata_t *meta) {
    buffer->tail_idx = NGX_HTTP_ML_DDOS_BUFFER_LIMIT(
        buffer->tail_idx + 1, NGX_HTTP_ML_DDOS_BUFFER_SIZE);

    uint32_t old_hash = hash_uint32(buffer->meta[buffer->tail_idx].uid);
    if (buffer->uid_table[old_hash] > 0)
        buffer->uid_table[old_hash]--;

    buffer->features[buffer->tail_idx] = *features;
    buffer->meta[buffer->tail_idx] = *meta;

    uint32_t new_hash = hash_uint32(meta->uid);
    buffer->uid_table[new_hash]++;
}

/**
 * Dumps the ring buffer into a contiguous buffer in linear chronological order
 *
 * +------------------------------+----------------------------------+
 * |     Second: copy this        |       First: copy this           |
 * |    (from start to tail)      |    (from tail + 1 to end)        |
 * +------------------------------+----------------------------------+
 * ^                              ^                                  ^
 * features                    tail_idx                        BUFFER_SIZE
 */
static ngx_inline mlds_features_t *
shm_buffer_features_dump(mlds_buffer_t *buffer, ngx_pool_t *pool) {
    mlds_features_t *features = ngx_palloc(pool, sizeof(buffer->features));
    if (!features)
        return NULL;

    const size_t size = sizeof(mlds_features_t);
    const size_t offset = NGX_HTTP_ML_DDOS_BUFFER_SIZE - 1 - buffer->tail_idx;

    memcpy(features, &buffer->features[buffer->tail_idx + 1], offset * size);
    memcpy(features + offset, buffer->features, (buffer->tail_idx + 1) * size);

    return features;
}

/// ===== DIRECTIVES ======

static ngx_int_t ngx_http_ml_ddos_init(ngx_conf_t *cf) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_conf_get_module_main_conf(cf, ngx_http_ml_ddos_module);
    if (!(mcf->shm_buffer = mlds_create_shm(cf, &ngx_http_ml_ddos_module)))
        return NGX_ERROR;

    ngx_http_core_main_conf_t *cmcf =
        ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);
    ngx_http_handler_pt *h =
        ngx_array_push(&cmcf->phases[NGX_HTTP_REWRITE_PHASE].handlers);
    if (!h)
        return NGX_ERROR;

    *h = ngx_http_ml_ddos_handler;

    return NGX_OK;
}

static char *ngx_http_ml_ddos_path(ngx_conf_t *cf, ngx_command_t *cmd,
                                   void *conf) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_conf_get_module_main_conf(cf, ngx_http_ml_ddos_module);
    ngx_str_t *value = cf->args->elts;
    mcf->model_path = value[1];

    return NGX_CONF_OK;
}

static ngx_inline ngx_int_t is_param(ngx_str_t *restrict value,
                                     const ngx_str_t *prefix) {
    return value->len >= prefix->len &&
           ngx_strncmp(value->data, prefix->data, prefix->len) == 0;
}

static ngx_inline ngx_str_t get_value(ngx_str_t *restrict value,
                                      const ngx_str_t *prefix) {
    ngx_str_t result;
    result.len = value->len - prefix->len;
    result.data = value->data + prefix->len;
    return result;
}

static ngx_thread_pool_t *parse_thread_pool(ngx_conf_t *cf,
                                            ngx_str_t *restrict param,
                                            const ngx_str_t *prefix) {
    ngx_str_t name = get_value(param, prefix);
    ngx_thread_pool_t *tpool = ngx_thread_pool_add(cf, &name);
    if (!tpool) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                           LOG_PREFIX "invalid thread pool \"%V\"", &name);
    }
    return tpool;
}

static ngx_flag_t parse_mode(ngx_conf_t *cf, ngx_str_t *restrict param,
                             const ngx_str_t *prefix) {
    ngx_str_t value = get_value(param, prefix);

    static const ngx_str_t sampling = ngx_string("sampling");
    static const ngx_str_t strict = ngx_string("strict");

    if (value.len == sampling.len &&
        ngx_strncasecmp(value.data, sampling.data, sampling.len) == 0)
        return 1;
    else if (value.len == strict.len &&
             ngx_strncasecmp(value.data, strict.data, strict.len) == 0)
        return 0;

    ngx_conf_log_error(
        NGX_LOG_EMERG, cf, NGX_ERROR,
        LOG_PREFIX
        "invalid mode value \"%V\" (expected 'sampling' or 'strict')",
        &value);

    return NGX_CONF_UNSET;
}

static float parse_float(ngx_conf_t *cf, ngx_str_t *restrict param,
                         const ngx_str_t *prefix) {
    ngx_str_t value = get_value(param, prefix);

    ngx_int_t v = ngx_atofp(value.data, value.len, 3);
    if (v == NGX_ERROR || v <= 0 || v > 1000) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           LOG_PREFIX "invalid %V value \"%V\"", prefix,
                           &value);
        return NAN;
    }

    return v / 1000.0f;
}

static char *ngx_http_ml_ddos_enable(ngx_conf_t *cf, ngx_command_t *cmd,
                                     void *conf) {
    ngx_http_ml_ddos_loc_conf_t *lcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (ngx_strcasecmp(value[1].data, (u_char *)"on") == 0) {
        lcf->enabled = 1;
    } else if (ngx_strcasecmp(value[1].data, (u_char *)"off") == 0) {
        lcf->enabled = 0;
    } else {
        return NGX_CONF_ERROR;
    }

    static const ngx_str_t thread_str = ngx_string("thread=");
    static const ngx_str_t mode_str = ngx_string("mode=");
    static const ngx_str_t block_str = ngx_string("block=");
    static const ngx_str_t limit_str = ngx_string("limit=");

    for (ngx_uint_t i = 2; i < cf->args->nelts; i++) {
        if (is_param(&value[i], &thread_str)) {
            lcf->thread_pool = parse_thread_pool(cf, &value[i], &thread_str);
            if (!lcf->thread_pool)
                return NGX_CONF_ERROR;
        } else if (is_param(&value[i], &mode_str)) {
            if ((lcf->sampling = parse_mode(cf, &value[i], &mode_str)) ==
                NGX_CONF_UNSET)
                return NGX_CONF_ERROR;
        } else if (is_param(&value[i], &block_str)) {
            lcf->block_threshold = parse_float(cf, &value[i], &block_str);
            if (isnan(lcf->block_threshold))
                return NGX_CONF_ERROR;
        } else if (is_param(&value[i], &limit_str)) {
            lcf->limit_threshold = parse_float(cf, &value[i], &limit_str);
            if (isnan(lcf->limit_threshold))
                return NGX_CONF_ERROR;
        } else {
            ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                               LOG_PREFIX "unknown parameter \"%V\"",
                               &value[i]);
            return NGX_CONF_ERROR;
        }
    }

    return NGX_CONF_OK;
}

/// ===== HANDLER =====

static ngx_inline uint64_t get_current_nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static ngx_inline uint32_t get_header_count(ngx_http_request_t *const r) {
    uint32_t result = 0;

    ngx_list_part_t *part = &r->headers_in.headers.part;
    while (part) {
        result += part->nelts;
        part = part->next;
    }

    return result;
}

static ngx_inline ngx_uint_t get_http_version(ngx_uint_t version) {
    switch (version) {
    case NGX_HTTP_VERSION_11: return 1;
    case NGX_HTTP_VERSION_20: return 2;
#if (nginx_version >= 1025000)
    case NGX_HTTP_VERSION_30: return 3;
#endif
    default: return 0;
    }
}

static ngx_inline ngx_uint_t get_http_req_type(int type) {
    switch (type) {
    case NGX_HTTP_GET:
    case NGX_HTTP_HEAD: return 1;
    case NGX_HTTP_POST:
    case NGX_HTTP_PUT:
    case NGX_HTTP_DELETE: return 2;
    case NGX_HTTP_CONNECT:
    case NGX_HTTP_OPTIONS:
    case NGX_HTTP_TRACE:
    case NGX_HTTP_PATCH: return 3;
    default: return 0;
    }
}

static ngx_inline void parse_params(ngx_http_request_t *r,
                                    mlds_features_params_t *params) {
    ngx_http_headers_in_t *headers = &r->headers_in;
    params->uri_length = r->uri.len;
    params->args_length = r->args.len;
    params->header_count = get_header_count(r);
    params->content_length =
        headers->content_length ? headers->content_length_n : 0;
    params->connection_requests = r->connection->requests;
}

static ngx_inline void parse_flags(ngx_http_request_t *r,
                                   mlds_features_flags_t *flags) {
    ngx_http_headers_in_t *headers = &r->headers_in;

    flags->http_version = get_http_version(r->http_version);
    flags->http_req_type = get_http_req_type(r->method);

    flags->has_user_agent = !!headers->user_agent;
    flags->has_ua_browser = headers->msie || headers->msie6 || headers->opera ||
                            headers->gecko || headers->chrome ||
                            headers->safari || headers->konqueror;
    flags->has_accept = !!headers->accept;
    flags->has_accept_language = !!headers->accept_language;
    flags->has_accept_encoding = !!headers->accept_encoding;
    flags->has_content_type = !!headers->content_type;
#if (nginx_version >= 1023000)
    flags->has_cookies = !!headers->cookie;
#else
    flags->has_cookies = !!headers->cookies.nelts;
#endif
    flags->has_authorization = !!headers->authorization;
    flags->has_referer = !!headers->referer;
    flags->has_host = !!headers->host;
    flags->has_upgrade = !!headers->upgrade;
    flags->has_transfer_encoding = !!headers->transfer_encoding;

    flags->keepalive = r->keepalive;
    flags->quoted_uri = r->quoted_uri;
}

static uint32_t get_client_ipv4_as_uint32(ngx_http_request_t *r) {
    struct sockaddr *sockaddr = r->connection->sockaddr;
    uint32_t ipv4 = 0;

    struct sockaddr_in *sin;
    struct sockaddr_in6 *sin6;
    switch (sockaddr->sa_family) {
    case AF_INET:
        sin = (struct sockaddr_in *)sockaddr;
        ipv4 = ntohl(sin->sin_addr.s_addr);
        break;
    case AF_INET6:
        sin6 = (struct sockaddr_in6 *)sockaddr;
        if (IN6_IS_ADDR_V4MAPPED(&sin6->sin6_addr)) {
            ngx_memcpy(&ipv4, &sin6->sin6_addr.s6_addr[12], 4);
            ipv4 = ntohl(ipv4);
        } else {
            ipv4 = (uint32_t)ngx_murmur_hash2(sin6->sin6_addr.s6_addr, 16);
        }
        break;
    }

    return ipv4;
}

/// Caps the nanosecond delta at ~4.29 seconds to prevent overflow
static ngx_inline uint32_t uint32_overflow_max(uint64_t a) {
    return a > UINT32_MAX ? UINT32_MAX : (uint32_t)a;
}

typedef struct ngx_http_ml_ddos_task_ctx {
    ngx_http_ml_ddos_main_conf_t *mcf;
    ngx_http_ml_ddos_loc_conf_t *lcf;
    mlds_features_t *features_matrix;
    ngx_http_request_t *r;
    ngx_int_t rc;
} mlds_task_ctx_t;

static void ngx_http_ml_ddos_worker(void *data, ngx_log_t *log) {
    mlds_task_ctx_t *ctx = data;
    ngx_http_ml_ddos_main_conf_t *mcf = ctx->mcf;
    ngx_http_request_t *r = ctx->r;
    log = r->connection->log;

    OrtStatus *status = NULL;
#define ONNX_ASSERT(expr)  \
    if ((status = (expr))) \
        goto done;

#define NGX_HTTP_ML_DDOS_INPUT_DIM 2
#define NGX_HTTP_ML_DDOS_OUTPUT_DIM 1

    int64_t dims[NGX_HTTP_ML_DDOS_INPUT_DIM] = {NGX_HTTP_ML_DDOS_BUFFER_SIZE,
                                                8};
    OrtValue *input_tensor = NULL;
    ONNX_ASSERT(mcf->ort_api->CreateTensorWithDataAsOrtValue(
        mcf->ort_memory_info, ctx->features_matrix,
        sizeof(*ctx->features_matrix) * NGX_HTTP_ML_DDOS_BUFFER_SIZE, dims,
        NGX_HTTP_ML_DDOS_INPUT_DIM, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        &input_tensor));

    static const char *input_names[] = {"input"};
    static const char *output_names[] = {"output"};

    OrtValue *outputs = NULL;
    ONNX_ASSERT(mcf->ort_api->Run(mcf->ort_session, NULL, input_names,
                                  (const OrtValue *const *)&input_tensor,
                                  NGX_HTTP_ML_DDOS_OUTPUT_DIM, output_names,
                                  NGX_HTTP_ML_DDOS_OUTPUT_DIM, &outputs));

    float *output_prob = NULL;
    ONNX_ASSERT(
        mcf->ort_api->GetTensorMutableData(outputs, (void **)&output_prob));
    float attack_prob = *output_prob;

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, log, NGX_OK,
                  LOG_PREFIX "DDOS probability: %.6f", attack_prob);
#endif

    if (attack_prob > ctx->lcf->block_threshold)
        ctx->rc = NGX_HTTP_FORBIDDEN;
    else if (attack_prob > ctx->lcf->limit_threshold)
        ctx->rc = NGX_HTTP_TOO_MANY_REQUESTS;
    else
        ctx->rc = NGX_DECLINED;

done:
    if (status) {
        ngx_log_error(NGX_LOG_ERR, log, NGX_ERROR, LOG_PREFIX "ONNX error: %s",
                      mcf->ort_api->GetErrorMessage(status));
        mcf->ort_api->ReleaseStatus(status);
        ctx->rc = NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    if (outputs)
        mcf->ort_api->ReleaseValue(outputs);
    if (input_tensor)
        mcf->ort_api->ReleaseValue(input_tensor);

#undef ONNX_ASSERT
}

static void ngx_http_ml_ddos_worker_done(ngx_event_t *ev) {
    mlds_task_ctx_t *ctx = ev->data;

    if (ctx->rc == NGX_DECLINED)
        ctx->r->phase_handler++;

    ngx_http_finalize_request(ctx->r, ctx->rc);
}

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_get_module_main_conf(r, ngx_http_ml_ddos_module);

    NGX_ASSERT(mcf->ort_api && mcf->ort_session, r->connection->log);

    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_http_get_module_loc_conf(r, ngx_http_ml_ddos_module);
    if (!lcf->enabled)
        return NGX_DECLINED;

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, NGX_OK,
                  LOG_PREFIX
                  "thread pool %p; sampling: %d; block %.3f; limit %.3f",
                  lcf->thread_pool, lcf->sampling, lcf->block_threshold,
                  lcf->limit_threshold);

    ngx_table_elt_t *h = ngx_list_push(&r->headers_out.headers);
    if (h) {
        ngx_str_set(&h->key, "X-HTTP-ML-DDOS");
        ngx_str_set(&h->value, "Enabled");
        h->hash = 1;
    }
#endif

    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *)mcf->shm_buffer->shm.addr;
    mlds_buffer_t *buffer = mcf->shm_buffer->data;

    mlds_metadata_t meta;
    meta.timestamp = get_current_nanos();
    meta.uid = get_client_ipv4_as_uint32(r);

    mlds_features_t features;
    mlds_features_params_t *params = &features.params;
    mlds_features_flags_t *flags = &params->flags;

    parse_params(r, params);
    parse_flags(r, flags);

    /// In sampling mode try to lock, if failed, skip
    if (lcf->sampling) {
        if (!ngx_shmtx_trylock(&shpool->mutex))
            return NGX_DECLINED;
    } else {
        ngx_shmtx_lock(&shpool->mutex);
    }

    params->ns_delta =
        uint32_overflow_max(meta.timestamp - shm_buffer_last_timestamp(buffer));
    params->ip_request_count = shm_buffer_uid_count(buffer, meta.uid);

    shm_buffer_push_element(buffer, &features, &meta);
    mlds_features_t *features_matrix =
        shm_buffer_features_dump(buffer, r->pool);

    ngx_shmtx_unlock(&shpool->mutex);

    if (!features_matrix)
        return NGX_HTTP_INTERNAL_SERVER_ERROR;

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, NGX_OK,
                  LOG_PREFIX "features:\n"
                             "\tns_delta: %ul\n"
                             "\tip_request_count: %ul\n"
                             "\turi_length: %ul\n"
                             "\targs_length: %ul\n"
                             "\theader_count: %ul\n"
                             "\tcontent_length: %ul\n"
                             "\tconnection_requests: %ul\n"
                             "\tflags: %ul\n",
                  features.data[0], features.data[1], features.data[2],
                  features.data[3], features.data[4], features.data[5],
                  features.data[6], features.data[7]);
#endif

    ngx_thread_task_t *task = ngx_thread_task_alloc(r->pool, sizeof(*task));
    if (!task)
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    mlds_task_ctx_t *ctx = task->ctx;

    ctx->mcf = mcf;
    ctx->lcf = lcf;
    ctx->features_matrix = features_matrix;
    ctx->r = r;

    if (lcf->thread_pool) {
        task->handler = ngx_http_ml_ddos_worker;
        task->event.handler = ngx_http_ml_ddos_worker_done;
        task->event.data = ctx;

        r->main->count++;
        ngx_int_t thread_status = ngx_thread_task_post(lcf->thread_pool, task);
        if (thread_status != NGX_OK) {
            r->main->count--;
            ngx_log_error(NGX_LOG_ERR, r->connection->log, thread_status,
                          LOG_PREFIX "Failed to add a task to the thread pool");
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }

        return NGX_AGAIN;
    } else {
        ngx_http_ml_ddos_worker(ctx, r->connection->log);
        return ctx->rc;
    }
}
