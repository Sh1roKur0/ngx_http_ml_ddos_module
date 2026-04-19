#include <ngx_core.h>
#include <ngx_http.h>
#include <onnxruntime_c_api.h>

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
} ngx_http_ml_ddos_loc_conf_t;

static const OrtApi *ort_api = NULL;
static OrtEnv *ort_env = NULL;
static OrtSessionOptions *ort_session_options = NULL;
static OrtSession *ort_session = NULL;

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

    {ngx_string("ml_ddos"),             // directive
     NGX_HTTP_LOC_CONF | NGX_CONF_FLAG, // location context
     ngx_http_ml_ddos_enable,           // configuration setup function
     NGX_HTTP_LOC_CONF_OFFSET,          // local offset
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
    return ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_loc_conf_t));
}

static char *ngx_http_ml_ddos_merge_loc_conf(ngx_conf_t *cf, void *parent,
                                             void *child) {
    ngx_http_ml_ddos_loc_conf_t *prev = parent;
    ngx_http_ml_ddos_loc_conf_t *conf = child;

    ngx_conf_merge_value(conf->enabled, prev->enabled, 0);

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
                                     &ort_env))) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "CreateEnv failed: %s",
                      ort_api->GetErrorMessage(status));
        goto error_env;
    }
    if ((status = ort_api->CreateSessionOptions(&ort_session_options))) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "CrateSessionOptions failed: %s",
                      ort_api->GetErrorMessage(status));
        goto error_options;
    }
    if ((status = ort_api->CreateSession(ort_env, (const char *)model_path_cstr,
                                         ort_session_options, &ort_session))) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, NGX_ERROR,
                      LOG_PREFIX "CreateSession failed: %s",
                      ort_api->GetErrorMessage(status));
        goto error_session;
    }

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, NGX_OK,
                  LOG_PREFIX "initialized with model: %V", &mcf->model_path);

    return NGX_OK;

error_session:
    ort_api->ReleaseSessionOptions(ort_session_options);
error_options:
    ort_api->ReleaseEnv(ort_env);
error_env:
    ort_api->ReleaseStatus(status);
    return NGX_ERROR;
}

static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle) {
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
    ngx_http_core_main_conf_t *cmcf;
    ngx_http_handler_pt *h;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_REWRITE_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

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

    return NGX_CONF_OK;
}

/// ===== HANDLER =====

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_http_get_module_loc_conf(r, ngx_http_ml_ddos_module);
    if (!lcf->enabled)
        return NGX_DECLINED;

#if NGX_DEBUG
    ngx_table_elt_t *h = ngx_list_push(&r->headers_out.headers);
    if (!h)
        return NGX_ERROR;

    ngx_str_set(&h->key, "X-HTTP-ML-DDOS");
    ngx_str_set(&h->value, "Enabled");
    h->hash = 1;
#endif

    NGX_ASSERT(ort_api && ort_session, r->connection->log);

    ngx_str_t client_ip = r->connection->addr_text;
    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, NGX_OK,
                  LOG_PREFIX "IP %V requested URI \"%V\"", &client_ip, &r->uri);

    return NGX_DECLINED;
}
