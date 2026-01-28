project(
    'ngx_http_ml_ddos_module',
    'c',
    default_options: [
        'c_std=c11',
        'b_lundef=false',
        'b_asneeded=false',
        'b_ndebug=if-release',
        'buildtype=release',
        'warning_level=1',
    ],
)

deps = [
    dependency('nginx', required: true),
    dependency('onnxruntime', required: true),
]
srcs = files('src/main.c')

shared_library(
    meson.project_name(),
    sources: srcs,
    dependencies: deps,
    install: true,
)
