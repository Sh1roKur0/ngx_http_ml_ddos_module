#include <ngx_core.h>
#include <ngx_http.h>
#include <onnxruntime_c_api.h>

#define LOG_PREFIX "ML_DDOS: "
#ifndef NDEBUG
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
    const OrtApi *api;
    OrtEnv *env;
} ngx_http_ml_ddos_main_conf_t;

typedef struct {
    OrtSession *session;
} ngx_http_ml_ddos_loc_conf_t;

static char *ngx_http_ml_ddos_directive(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf);
static void *ngx_http_ml_ddos_create_main_conf(ngx_conf_t *cf);
static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf);
static void ngx_http_ml_ddos_cleanup_loc_conf(void *data);
static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r);

static ngx_command_t ngx_http_ml_ddos_commands[] = {

    {ngx_string("ngx_http_ml_ddos"),                       // directive
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1 | NGX_CONF_NOARGS, // location context
     ngx_http_ml_ddos_directive, // configuration setup function
     NGX_HTTP_LOC_CONF_OFFSET,   // local offset
     0,                          // configuration offset
     NULL},

    ngx_null_command};

static ngx_http_module_t ngx_http_ml_ddos_module_ctx = {
    NULL,                              // preconfiguration
    NULL,                              // postconfiguration
    ngx_http_ml_ddos_create_main_conf, // create main configuration
    NULL,                              // init main configuration
    NULL,                              // create server configuration
    NULL,                              // merge server configuration
    ngx_http_ml_ddos_create_loc_conf,  // create local configuration
    NULL                               // merge local configuration
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
ngx_module_t *ngx_modules[] = {&ngx_http_ml_ddos_module, NULL};
char *ngx_module_names[] = {"ngx_http_ml_ddos_module", NULL};
char *ngx_module_order[] = {NULL};

static void ngx_http_ml_ddos_cleanup_main_conf(void *data) {
    ngx_http_ml_ddos_main_conf_t *mcf = data;
    mcf->api->ReleaseEnv(mcf->env);
    mcf->env = NULL;
}

static void *ngx_http_ml_ddos_create_main_conf(ngx_conf_t *cf) {
    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_main_conf_t));
    if (!mcf)
        return NULL;

    if (!(mcf->api = OrtGetApiBase()->GetApi(ORT_API_VERSION)))
        return NULL;
    if (mcf->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ml_ddos_env",
                            &mcf->env))
        return NULL;

    ngx_pool_cleanup_t *cln = ngx_pool_cleanup_add(cf->pool, 0);
    if (!cln)
        return NULL;

    cln->handler = ngx_http_ml_ddos_cleanup_main_conf;
    cln->data = mcf;

    return mcf;
}

typedef struct {
    ngx_http_ml_ddos_loc_conf_t *lcf;
    ngx_http_ml_ddos_main_conf_t *mcf;
} ngx_http_ml_ddos_cleanup_ctx_t;

static void ngx_http_ml_ddos_cleanup_loc_conf(void *data) {
    ngx_http_ml_ddos_cleanup_ctx_t *ctx = data;

    if (ctx->lcf && ctx->mcf && ctx->lcf->session) {
        ctx->mcf->api->ReleaseSession(ctx->lcf->session);
        ctx->lcf->session = NULL;
    }
}

static void *ngx_http_ml_ddos_create_loc_conf(ngx_conf_t *cf) {
    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_pcalloc(cf->pool, sizeof(ngx_http_ml_ddos_loc_conf_t));
    if (!lcf)
        return NULL;

    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_conf_get_module_main_conf(cf, ngx_http_ml_ddos_module);
    if (!mcf)
        return NULL;

    ngx_pool_cleanup_t *cln =
        ngx_pool_cleanup_add(cf->pool, sizeof(ngx_http_ml_ddos_cleanup_ctx_t));
    if (!cln)
        return NULL;

    cln->handler = ngx_http_ml_ddos_cleanup_loc_conf;
    const ngx_http_ml_ddos_cleanup_ctx_t ctx = {lcf, mcf};
    ngx_memcpy(cln->data, &ctx, sizeof(ctx));

    return lcf;
}

static char *ngx_http_ml_ddos_directive(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf) {
    (void)cmd;

    ngx_http_ml_ddos_main_conf_t *mcf =
        ngx_http_conf_get_module_main_conf(cf, ngx_http_ml_ddos_module);
    NGX_ASSERT(mcf && mcf->api && mcf->env, cf->log);

    ngx_http_ml_ddos_loc_conf_t *lcf = conf;
    if (lcf->session)
        return "ngx_http_ml_ddos is already defined for this location";

    const char *model_path = "/etc/nginx/model.onnx";
    if (cf->args->nelts == 2) {
        ngx_str_t path = ((ngx_str_t *)cf->args->elts)[1];
        model_path = ngx_pcalloc(cf->pool, path.len + 1);
        if (!model_path)
            return NGX_CONF_ERROR;
        ngx_cpystrn((u_char *)model_path, path.data, path.len + 1);
    }

    ngx_file_info_t fi;
    if (ngx_file_info(model_path, &fi) != NGX_OK)
        return "Model file not found";

    OrtStatusPtr status;
#define CHECK_ORT(expr, err) \
    if ((status = expr))     \
    goto err

    ngx_log_debug(NGX_LOG_NOTICE, cf->log, NGX_OK,
                  LOG_PREFIX "Loading model from: %s", model_path);
    OrtSessionOptions *session_options;
    CHECK_ORT(mcf->api->CreateSessionOptions(&session_options), opt_err);
    CHECK_ORT(mcf->api->SetIntraOpNumThreads(session_options, 1), session_err);
    CHECK_ORT(
        mcf->api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL),
        session_err);
    CHECK_ORT(mcf->api->CreateSession(mcf->env, model_path, session_options,
                                      &lcf->session),
              session_err);
    mcf->api->ReleaseSessionOptions(session_options);

    ngx_http_core_loc_conf_t *clcf =
        ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    NGX_ASSERT(clcf, cf->log);
    clcf->handler = ngx_http_ml_ddos_handler;

    ngx_log_debug(NGX_LOG_NOTICE, cf->log, NGX_OK,
                  LOG_PREFIX "Module is being initialized!");

    return NGX_CONF_OK;

session_err:
    mcf->api->ReleaseSessionOptions(session_options);

opt_err:;
    const char *msg = mcf->api->GetErrorMessage(status);
    ngx_conf_log_error(NGX_LOG_EMERG, cf, NGX_ERROR,
                       LOG_PREFIX "ONNX Error: %s", msg);
    mcf->api->ReleaseStatus(status);
    return "Failed to create ONNX session";

#undef CHECK_ORT
}

static ngx_int_t ngx_http_ml_ddos_handler(ngx_http_request_t *r) {
    ngx_http_ml_ddos_loc_conf_t *lcf =
        ngx_http_get_module_loc_conf(r, ngx_http_ml_ddos_module);
    NGX_ASSERT(lcf && lcf->session, r->connection->log);

    ngx_str_t client_ip = r->connection->addr_text;
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, NGX_OK,
                  LOG_PREFIX "IP %V requested URI \"%V\"", &client_ip, &r->uri);

    return NGX_DECLINED;
}
