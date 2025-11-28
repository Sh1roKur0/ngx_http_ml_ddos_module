#include <sys/syscall.h>

int main(void) {
    (void)SYS_eventfd;
    return 0;
}
