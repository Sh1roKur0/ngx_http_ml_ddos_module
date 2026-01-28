#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <onnxruntime_c_api.h>

static const OrtApi *ort_api = NULL;
static OrtEnv *ort_env = NULL;
static OrtSession *ort_session = NULL;

static char *ngx_http_ml_ddos(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r);
static ngx_int_t ngx_http_ml_ddos_init_process(ngx_cycle_t *cycle);
static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle);

static ngx_command_t ngx_http_ml_ddos_commands[] = {

    {ngx_string("ngx_http_ml_ddos"),      // directive
     NGX_HTTP_LOC_CONF | NGX_CONF_NOARGS, // location context
     ngx_http_ml_ddos,                    // configuration setup function
     0, // No offset. Only one context is supported.
     0, // No offset when storing the module configuration on struct.
     NULL},

    ngx_null_command // command termination
};

// No context for the module
static ngx_http_module_t ngx_http_ml_ddos_module_ctx = {NULL};

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
ngx_module_t *ngx_modules[] = {&ngx_http_ml_ddos_module, NULL};
char *ngx_module_names[] = {"ngx_http_ml_ddos_module", NULL};
char *ngx_module_order[] = {NULL};

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
#ifndef NDEBUG
    ngx_str_t client_ip = r->connection->addr_text;
    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, 0,
                  "ML_DDOS Log: IP %V requested URI \"%V\"", &client_ip,
                  &r->uri);
#endif

    return NGX_DECLINED;
}

static ngx_int_t ngx_http_ml_ddos_init_process(ngx_cycle_t *cycle) {
#define ASSERT_ORT(expr) \
    if (expr)            \
    goto error

    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ASSERT_ORT(
        ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ml_ddos_env", &ort_env));

    OrtSessionOptions *session_options;
    ASSERT_ORT(ort_api->CreateSessionOptions(&session_options));
    ASSERT_ORT(ort_api->SetIntraOpNumThreads(session_options, 1));

    const char *model_path = "/etc/nginx/model.onnx";
    ASSERT_ORT(ort_api->CreateSession(ort_env, model_path, session_options,
                                      &ort_session));

    ort_api->ReleaseSessionOptions(session_options);

    ngx_log_error(NGX_LOG_NOTICE, cycle->log, 0,
                  "ML_DDOS: ONNX Session Initialized");
    return NGX_OK;
error:
    ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                  "ML_DDOS: Failed to load ONNX model");
    return NGX_ERROR;

#undef ASSERT_ORT
}

static void ngx_http_ml_ddos_exit_process(ngx_cycle_t *cycle) {
    if (ort_session)
        ort_api->ReleaseSession(ort_session);
    if (ort_env)
        ort_api->ReleaseEnv(ort_env);
}

static char *ngx_http_ml_ddos(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {

    ngx_http_core_loc_conf_t *clcf;

    clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = ngx_http_ml_ddos_handler;

#ifndef NDEBUG
    ngx_conf_log_error(NGX_LOG_NOTICE, cf, 0,
                       "ML_DDOS module is being initialized!");
#endif

    return NGX_CONF_OK;
}
