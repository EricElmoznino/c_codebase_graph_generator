#include <stdio.h>
#include "math_utils.h"
#include "string_utils.h"

int main(void) {
    int x = 2, y = 3;
    int sum     = add(x, y);
    int product = multiply(x, y);

    printf("Sum: %d\n", sum);
    printf("Product: %d\n", product);

    greet("World");
    return 0;
}
