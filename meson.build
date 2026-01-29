project(
    'ngx_http_ml_ddos_module',
    'c',
    version: '0.1.2',
    meson_version: '>= 1.3.0',
    default_options: [
        'c_std=c11',
        'b_lundef=false',
        'b_asneeded=false',
        'b_ndebug=if-release',
        'buildtype=release',
        'warning_level=1',
        'prefix=/etc/nginx/',
    ],
    license: 'BSD-3-Clause',
    license_files: 'LICENSE',
)

fs = import('fs')
py3 = import('python').find_installation()
ngx_bin = find_program(
    'nginx',
    dirs: ['/usr/sbin', '/usr/local/sbin', '/opt/nginx/bin'],
    required: true,
)

cp_script = 'import shutil; shutil.copytree("@0@", "@1@", dirs_exist_ok=True)'
run_command(
    py3,
    '-c',
    cp_script.format(
        get_option('nginx_src'),
        meson.current_build_dir() / 'nginx_src',
    ),
    check: true,
)
ngx_src = meson.current_build_dir() / 'nginx_src'
ngx_src_relative = '.' / fs.relative_to(ngx_src, meson.current_source_dir())

if not fs.exists(ngx_src / 'objs')
    message('Configurating nginx\'s options...')
    ngx_flags = run_command(ngx_bin, '-V', check: true).stderr().strip()
    configure_script = '''
import subprocess, sys, os
import subprocess, sys, os

target_dir = sys.argv[1]
raw_output = sys.argv[2]

flags = [arg for arg in raw_output.split() if arg.startswith("--") and "module" not in arg and "opt" not in arg]

os.chdir(target_dir)
subprocess.run(["./configure"] + flags, check=True)
'''
    run_command(py3, '-c', configure_script, ngx_src, ngx_flags, check: true)
endif

ngx_includes = [
    ngx_src_relative,
    ngx_src_relative / 'src/core',
    ngx_src_relative / 'src/event',
    ngx_src_relative / 'src/event/modules',
    ngx_src_relative / 'src/event/quic',
    ngx_src_relative / 'src/http',
    ngx_src_relative / 'src/http/modules',
    ngx_src_relative / 'objs',
]
if target_machine.system() == 'windows'
    ngx_includes += ngx_src_relative / 'src/os/win32'
else
    ngx_includes += ngx_src_relative / 'src/os/unix'
endif

deps = [
    dependency('onnxruntime', required: true),
]
srcs = files('module.c')

shared_library(
    meson.project_name(),
    sources: srcs,
    dependencies: deps,
    include_directories: ngx_includes,
    install: true,
    install_dir: 'modules',
    name_prefix: '',
)
