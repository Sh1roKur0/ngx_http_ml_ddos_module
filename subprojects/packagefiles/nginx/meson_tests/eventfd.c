#include <sys/eventfd.h>

int main(void) {
    (void)eventfd(0, 0);
    return 0;
}
