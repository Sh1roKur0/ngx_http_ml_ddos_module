project(
  'nginx',
  'c',
  version: '1.29.1',
  license: 'BSD-2-Clause',
  license_files: 'LICENSE',
)

cc = meson.get_compiler('c')
system = target_machine.system()

inc_dirs = []
inc_dirs += include_directories('.')
inc_dirs += include_directories('src/core')
inc_dirs += include_directories('src/event')
inc_dirs += include_directories('src/event/modules')
inc_dirs += include_directories('src/event/quic')
inc_dirs += include_directories('src/http')
inc_dirs += include_directories('src/http/modules')
if system == 'windows'
  inc_dirs += include_directories('src/os/win32')
else
  inc_dirs += include_directories('src/os/unix')
endif

ngx_macro_value = '''
#ifndef NGX_@0@
#define NGX_@0@ @1@
#endif
'''
ngx_macro = ngx_macro_value.format('@0@', 1)
ngx_have = ngx_macro.format('HAVE_@0@')

conf_data = configuration_data()

header_checks = [
  ['UNISTD_H', 'unistd.h'],
  ['INTTYPES_H', 'inttypes.h'],
  ['LIMITS_H', 'limits.h'],
  ['CRYPT_H', 'crypt.h'],
  ['SYS_PARAM_H', 'sys/param.h'],
  ['SYS_MOUNT_H', 'sys/mount.h'],
  ['SYS_STATVFS_H', 'sys/statvfs.h'],
  ['SYS_PRCTL_H', 'sys/prctl.h'],
  ['SYS_VFS_H', 'sys/vfs.h'],
]
foreach header : header_checks
  conf_data.set(
    header[0],
    cc.has_header(header[1]) ? ngx_have.format(header[0]) : '',
  )
endforeach

ngx_system_names = {
  'linux': 'LINUX',
  'android': 'LINUX',
  'gnu': 'LINUX',
  'cygwin': 'LINUX',
  'freebsd': 'FREEBSD',
  'sunos': 'SOLARIS',
  'windows': 'WIN32',
  'darwin': 'DARWIN',
}
conf_data.set(
  'SYSTEM',
  system in ngx_system_names ? ngx_macro.format(ngx_system_names[system]) : '',
)

configure_file(
  input: 'ngx_auto_headers.h.in',
  output: 'ngx_auto_headers.h',
  configuration: conf_data,
)

conf_data = configuration_data()

file_tests = [
  'gcc_atomic',
  'c99_varmac',
  'epoll',
  'epollrdhup',
  'epollexclusive',
  'eventfd',
  'sys_eventfd',
]
foreach file_test : file_tests
  conf_data.set(
    file_test,
    cc.compiles(files('meson_tests/@0@.c'.format(file_test))).to_int(),
  )
endforeach

configure_file(
  input: 'ngx_auto_config.h.in',
  output: 'ngx_auto_config.h',
  configuration: conf_data,
)

nginx_dep = declare_dependency(include_directories: inc_dirs)
