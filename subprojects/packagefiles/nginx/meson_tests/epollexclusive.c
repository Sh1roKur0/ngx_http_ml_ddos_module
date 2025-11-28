#include <stddef.h>
#include <sys/epoll.h>

int main(void) {
    int efd = 0, fd = 0;
    struct epoll_event ee;
    ee.events = EPOLLIN | EPOLLEXCLUSIVE;
    ee.data.ptr = NULL;
    epoll_ctl(efd, EPOLL_CTL_ADD, fd, &ee);
    return 0;
}
