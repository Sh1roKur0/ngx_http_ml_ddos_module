#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_thread_pool.h>
#include <onnxruntime_c_api.h>

#include <ctype.h>
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

typedef struct {
    ngx_str_t model_path;
} ngx_http_ml_ddos_main_conf_t;

typedef struct {
    ngx_flag_t enabled;
    ngx_thread_pool_t *thread_pool;
    float block_threshold;
    float limit_threshold;
} ngx_http_ml_ddos_loc_conf_t;

static const OrtApi *ort_api = NULL;
static OrtEnv *ort_env = NULL;
static OrtSessionOptions *ort_session_options = NULL;
static OrtSession *ort_session = NULL;
static OrtAllocator *ort_allocator = NULL;
static OrtMemoryInfo *ort_memory_info = NULL;

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
    if (conf->thread_pool == NULL) {
        conf->thread_pool = prev->thread_pool;
    }
    if (isnan(conf->block_threshold)) {
        conf->block_threshold =
            isnan(prev->block_threshold) ? 0.85f : prev->block_threshold;
    }

    if (isnan(conf->limit_threshold)) {
        conf->limit_threshold =
            isnan(prev->limit_threshold) ? 0.65f : prev->limit_threshold;
    }

    return NGX_CONF_OK;
}

/// ===== PROCESS ======

static ngx_int_t ngx_http_ml_ddos_init_process(ngx_cycle_t *cycle) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_cycle_get_module_main_conf(cycle, ngx_http_ml_ddos_module);
    if (mcf == NULL || mcf->model_path.len == 0) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "model path not configured");
        return NGX_ERROR;
    }

    u_char *model_path_cstr = ngx_pstrdup(cycle->pool, &mcf->model_path);
    ngx_file_info_t fi;
    if (!model_path_cstr ||
        ngx_file_info((const char *)model_path_cstr, &fi) == NGX_FILE_ERROR) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno,
                      LOG_PREFIX "model not found: %s", model_path_cstr);
        return NGX_ERROR;
    }

    if (!(ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION))) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "failed to get ONNX API");
        return NGX_ERROR;
    }

    OrtStatus *status;
    if ((status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ngx_ml_ddos",
                                     &ort_env)))
        goto error_env;
    if ((status = ort_api->CreateSessionOptions(&ort_session_options)))
        goto error_options;
    if ((status = ort_api->SetIntraOpNumThreads(ort_session_options, 1)))
        goto error_options;
    if ((status = ort_api->SetInterOpNumThreads(ort_session_options, 1)))
        goto error_options;
    if ((status = ort_api->CreateSession(ort_env, (const char *)model_path_cstr,
                                         ort_session_options, &ort_session)))
        goto error_session;
    if ((status = ort_api->GetAllocatorWithDefaultOptions(&ort_allocator)))
        goto error_memory;
    if ((status = ort_api->CreateCpuMemoryInfo(
             OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info)))
        goto error_memory;

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, NGX_OK,
                  LOG_PREFIX "initialized with model: %V", &mcf->model_path);

    return NGX_OK;

error_memory:
    ort_api->ReleaseSession(ort_session);
error_session:
    ort_api->ReleaseSessionOptions(ort_session_options);
error_options:
    ort_api->ReleaseEnv(ort_env);
error_env:
    ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                  LOG_PREFIX "ONNX initialize failed: %s",
                  ort_api->GetErrorMessage(status));
    ort_api->ReleaseStatus(status);
    return NGX_ERROR;
}

static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle) {
    NGX_ASSERT(ort_session && ort_session_options && ort_env && ort_allocator &&
                   ort_memory_info,
               cycle->log);

    if (ort_allocator) {
        ort_api->ReleaseAllocator(ort_allocator);
        ort_allocator = NULL;
    }

    if (ort_memory_info) {
        ort_api->ReleaseMemoryInfo(ort_memory_info);
        ort_memory_info = NULL;
    }

    if (ort_session) {
        ort_api->ReleaseSession(ort_session);
        ort_session = NULL;
    }

    if (ort_session_options) {
        ort_api->ReleaseSessionOptions(ort_session_options);
        ort_session_options = NULL;
    }

    if (ort_env) {
        ort_api->ReleaseEnv(ort_env);
        ort_env = NULL;
    }

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, NGX_OK,
                  LOG_PREFIX "ONNX runtime shutdown");
}

/// ===== DIRECTIVES ======

static ngx_int_t ngx_http_ml_ddos_init(ngx_conf_t *cf) {
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
    static const ngx_str_t block_str = ngx_string("block=");
    static const ngx_str_t limit_str = ngx_string("limit=");

#define GET_NAME(str)                   \
    ngx_str_t name;                     \
    name.len = value[i].len - str.len;  \
    name.data = value[i].data + str.len

    for (ngx_uint_t i = 2; i < cf->args->nelts; i++) {
        if (ngx_strncmp(value[i].data, thread_str.data, thread_str.len) == 0) {
            GET_NAME(thread_str);

            lcf->thread_pool = ngx_thread_pool_add(cf, &name);
            if (lcf->thread_pool == NULL) {
                ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                                   LOG_PREFIX "invalid thread pool \"%V\"",
                                   &name);
                return NGX_CONF_ERROR;
            }
        } else if (ngx_strncmp(value[i].data, block_str.data, block_str.len) ==
                   0) {
            GET_NAME(block_str);
            ngx_int_t v = ngx_atofp(name.data, name.len, 3);
            if (v == NGX_ERROR || v <= 0) {
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                                   LOG_PREFIX "invalid block value \"%V\"",
                                   &value[i]);
                return NGX_CONF_ERROR;
            }
            lcf->block_threshold = v / 1000.0f;
        } else if (ngx_strncmp(value[i].data, limit_str.data, limit_str.len) ==
                   0) {
            GET_NAME(limit_str);
            ngx_int_t v = ngx_atofp(name.data, name.len, 3);
            if (v == NGX_ERROR || v <= 0) {
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                                   LOG_PREFIX "invalid limit value \"%V\"",
                                   &value[i]);
                return NGX_CONF_ERROR;
            }
            lcf->limit_threshold = v / 1000.0f;
        } else {
            ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                               LOG_PREFIX "unknown parameter \"%V\"",
                               &value[i]);
            return NGX_CONF_ERROR;
        }
    }

    if (lcf->limit_threshold >= lcf->block_threshold) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           LOG_PREFIX "limit must be less than block");
        return NGX_CONF_ERROR;
    }

#undef GET_NAME

    return NGX_CONF_OK;
}

/// ===== HANDLER =====

typedef struct {
    float block_threshold;
    float limit_threshold;
    ngx_http_request_t *r;
    ngx_int_t rc;
} ngx_http_ml_ddos_task_ctx_t;

static void ngx_http_ml_ddos_worker(void *data, ngx_log_t *log) {
    ngx_http_ml_ddos_task_ctx_t *ctx = data;
    ngx_http_request_t *r = ctx->r;
    log = r->connection->log;

    float intensity = (float)r->connection->requests;
    float request_length = (float)r->request_length;
    float url_length = (float)r->request_line.len;
    float args_length = (float)r->args.len + !!r->args.len;
    float ua_length = r->headers_in.user_agent
                          ? (float)r->headers_in.user_agent->value.len
                          : 0.0f;

    ngx_time_t *tp = ngx_timeofday();
    ngx_msec_t ms =
        (tp->sec - r->start_sec) * 1000 + (tp->msec - r->start_msec);
    float request_time = ms > 0 ? (float)ms / 1000.0f : 0.0f;

    float special_chars = 0.0f;
    for (size_t i = 0; i < r->args.len; i++)
        if (!isalnum((unsigned char)r->args.data[i]))
            special_chars++;

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, log, NGX_OK,
                  LOG_PREFIX "PARAMS:\n"
                             "\tintensivity:\t%f\n"
                             "\trequest_length:\t%f\n"
                             "\trequest_time:\t%.6f\n"
                             "\turl_length:\t%f\n"
                             "\targs_length:\t%f\n"
                             "\tspecial_chars:\t%f\n"
                             "\tua_length:\t%f",
                  intensity, request_length, request_time, url_length,
                  args_length, special_chars, ua_length);
#endif

    float features[7];
    features[0] = intensity;
    features[1] = request_length;
    features[2] = request_time;
    features[3] = url_length;
    features[4] = args_length;
    features[5] = special_chars;
    features[6] = ua_length;

    OrtStatus *status = NULL;
#define ONNX_ASSERT(expr)      \
    do {                       \
        if ((status = (expr))) \
            goto done;         \
    } while (0)

    int64_t dims[2] = {1, 7};
    OrtValue *input_tensor = NULL;
    ONNX_ASSERT(ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, features, sizeof(features), dims, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char *input_names[] = {"features"};
    const char *output_names[] = {"label", "probabilities"};

    OrtValue *outputs[2] = {NULL, NULL};
    ONNX_ASSERT(ort_api->Run(ort_session, NULL, input_names,
                             (const OrtValue *const *)&input_tensor, 1,
                             output_names, 2, outputs));

    ONNXType out_type;
    ONNX_ASSERT(ort_api->GetValueType(outputs[1], &out_type));

    if (out_type != ONNX_TYPE_SEQUENCE) {
        ctx->rc = NGX_HTTP_INTERNAL_SERVER_ERROR;
        goto done;
    }

    size_t seq_len = 0;
    ONNX_ASSERT(ort_api->GetValueCount(outputs[1], &seq_len));
    if (seq_len == 0) {
        ctx->rc = NGX_HTTP_INTERNAL_SERVER_ERROR;
        goto done;
    }

    OrtValue *seq_elem = NULL;
    ONNX_ASSERT(ort_api->GetValue(outputs[1], 0, ort_allocator, &seq_elem));

    int64_t key = 1;
    OrtValue *tensor = NULL;
    ONNX_ASSERT(ort_api->GetValue(seq_elem, key, ort_allocator, &tensor));

    float *probs = NULL;
    ONNX_ASSERT(ort_api->GetTensorMutableData(tensor, (void **)&probs));

    float attack_prob = probs[1];

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, log, NGX_OK,
                  LOG_PREFIX "DDOS probability: %.6f", attack_prob);
#endif

    if (attack_prob > ctx->block_threshold)
        ctx->rc = NGX_HTTP_FORBIDDEN;
    else if (attack_prob > ctx->limit_threshold)
        ctx->rc = NGX_HTTP_TOO_MANY_REQUESTS;
    else
        ctx->rc = NGX_DECLINED;

done:
    if (status) {
        ngx_log_error(NGX_LOG_ERR, log, NGX_ERROR, LOG_PREFIX "ONNX error: %s",
                      ort_api->GetErrorMessage(status));
        ort_api->ReleaseStatus(status);
        ctx->rc = NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    if (input_tensor)
        ort_api->ReleaseValue(input_tensor);
    if (outputs[0])
        ort_api->ReleaseValue(outputs[0]);
    if (outputs[1])
        ort_api->ReleaseValue(outputs[1]);
    if (tensor)
        ort_api->ReleaseValue(tensor);
    if (seq_elem)
        ort_api->ReleaseValue(seq_elem);
}

static void ngx_http_ml_ddos_worker_done(ngx_event_t *ev) {
    ngx_http_ml_ddos_task_ctx_t *ctx = ev->data;

    if (ctx->rc == NGX_DECLINED)
        ctx->r->phase_handler++;

    ngx_http_finalize_request(ctx->r, ctx->rc);
}

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
    NGX_ASSERT(ort_api && ort_session, r->connection->log);

    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_http_get_module_loc_conf(r, ngx_http_ml_ddos_module);
    if (!lcf->enabled)
        return NGX_DECLINED;

#if NGX_DEBUG
    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, NGX_OK,
                  LOG_PREFIX "thread pool %p; block %.3f; limit %.3f",
                  lcf->thread_pool, lcf->block_threshold, lcf->limit_threshold);

    ngx_table_elt_t *h = ngx_list_push(&r->headers_out.headers);
    if (!h)
        return NGX_ERROR;

    ngx_str_set(&h->key, "X-HTTP-ML-DDOS");
    ngx_str_set(&h->value, "Enabled");
    h->hash = 1;
#endif

    ngx_thread_task_t *task =
        ngx_thread_task_alloc(r->pool, sizeof(ngx_http_ml_ddos_task_ctx_t));
    if (!task)
        return NGX_HTTP_INTERNAL_SERVER_ERROR;

    ngx_http_ml_ddos_task_ctx_t *ctx = task->ctx;
    ctx->block_threshold = lcf->block_threshold;
    ctx->limit_threshold = lcf->limit_threshold;
    ctx->r = r;

    if (lcf->thread_pool) {
        task->handler = ngx_http_ml_ddos_worker;
        task->event.handler = ngx_http_ml_ddos_worker_done;
        task->event.data = ctx;

        ngx_int_t thread_status = ngx_thread_task_post(lcf->thread_pool, task);
        if (thread_status != NGX_OK) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, thread_status,
                          LOG_PREFIX "Failed to add a task to the thread pool");
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }
        r->main->count++;

        return NGX_AGAIN;
    } else {
        ngx_http_ml_ddos_worker(ctx, r->connection->log);
        return ctx->rc;
    }
}
