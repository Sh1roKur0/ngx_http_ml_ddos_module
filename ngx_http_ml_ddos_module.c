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
} ngx_http_ml_ddos_loc_conf_t;

static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_ml_ddos_merge_loc_conf(ngx_conf_t *cf, void *parent,
                                             void *child);

static char *ngx_http_ml_ddos_directive(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf);
static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r);

static ngx_command_t ngx_http_ml_ddos_commands[] = {

    {ngx_string("ngx_http_ml_ddos"),     // directive
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1, // location context
     ngx_http_ml_ddos_directive,         // configuration setup function
     NGX_HTTP_LOC_CONF_OFFSET,           // local offset
     0,                                  // configuration offset
     NULL},

    ngx_null_command};

static ngx_http_module_t ngx_http_ml_ddos_module_ctx = {
    NULL,                             // preconfiguration
    NULL,                             // postconfiguration
    NULL,                             // create main configuration
    NULL,                             // init main configuration
    NULL,                             // create server configuration
    NULL,                             // merge server configuration
    ngx_http_ml_ddos_create_loc_conf, // create local configuration
    ngx_http_ml_ddos_merge_loc_conf   // merge local configuration
};

ngx_module_t ngx_http_ml_ddos_module = {
    NGX_MODULE_V1,
    &ngx_http_ml_ddos_module_ctx, // module context
    ngx_http_ml_ddos_commands,    // module directives
    NGX_HTTP_MODULE,              // module type
    NULL,                         // init master
    NULL,                         // init module
    NULL,                         // init process
    NULL,                         // init thread
    NULL,                         // exit thread
    NULL,                         // exit process
    NULL,                         // exit master
    NGX_MODULE_V1_PADDING};

/// ===== CONFIG =====

static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf) {
    return ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_loc_conf_t));
}

static char *ngx_http_ml_ddos_merge_loc_conf(ngx_conf_t *cf, void *parent,
                                             void *child) {
    ngx_http_ml_ddos_loc_conf_t *prev = parent, *conf = child;
    ngx_conf_merge_str_value(conf->model_path, prev->model_path, "");
    return NGX_CONF_OK;
}

/// ===== MODULE ======

static char *ngx_http_ml_ddos_directive(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf) {
    ngx_http_ml_ddos_loc_conf_t *mlcf = conf;
    ngx_str_t *value = cf->args->elts;
    mlcf->model_path = value[1];

    ngx_http_core_loc_conf_t *clcf =
        ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = ngx_http_ml_ddos_handler;

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
    ngx_http_ml_ddos_loc_conf_t *mlcf;
    mlcf = ngx_http_get_module_loc_conf(r, ngx_http_ml_ddos_module);

    ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                  LOG_PREFIX "Model path: %V", &mlcf->model_path);

    ngx_str_t client_ip = r->connection->addr_text;
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, NGX_OK,
                  LOG_PREFIX "IP %V requested URI \"%V\"", &client_ip, &r->uri);

    return NGX_DECLINED;
}
