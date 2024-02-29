from tinygrad.tensor import Tensor as tn
from tinygrad.helpers import getenv


class Linear:
    def __init__(self, input_size, output_size):
        # Initialize weights and bias
        self.weights = tn.randn(input_size, output_size)
        self.bias = tn.randn(1, output_size)

        if getenv("DEBUG") > 0:
            print(
                f"Initialized Linear layer with weights: {self.weights.shape} and bias: {self.bias.shape}"
            )

    def forward(self, inputs):
        # Perform dot product of inputs and weights
        weighted_sum = tn.dot(inputs, self.weights)
        if getenv("DEBUG") > 0:
            print(f"\t-- {inputs.shape} * {self.weights.shape} = {weighted_sum.shape}")

        # Add the bias
        output = weighted_sum + self.bias
        if getenv("DEBUG") > 0:
            print(f"\t-- {weighted_sum.shape} + {self.bias.shape} = {output.shape}")

        return output


if __name__ == "__main__":
    # Create a Linear layer
    linear = Linear(5, 1)

    # Create some random inputs
    inputs = tn.randn(1, 5)

    # Forward pass
    output = linear.forward(inputs)

    if getenv("DEBUG") > 0:
        print(f"Output shape: {output.shape}, Output: {output.numpy()}")

    from tinygrad.device import Device, Compiled
    from tinygrad.realize import create_schedule
    from tinygrad.ops import LoadOps
    from tinygrad.codegen.linearizer import Linearizer
    from tinygrad.features.search import time_linearizer, bufs_from_lin
    from tinygrad.shape.symbolic import sym_infer
    from tinygrad.helpers import ansilen
    from tinygrad.features.search import get_linearizer_actions

    from tinygrad.features.graph import graph_uops

    seen = set()

    create_schedule([linear.forward(tn.empty(1, 5)).lazydata], seen)

    # the device we are optimizing for
    device: Compiled = Device[Device.DEFAULT]
    if getenv("DEBUG") > 0:
        print(f"optimizing for {Device.DEFAULT}")

    out = linear.forward(tn.randn(1, 5))
    sched = create_schedule([out.lazydata], seen)
    sched = [x for x in sched if x.ast.op not in LoadOps]

    total_tm = 0
    running_gflops = 0
    for i, kernel in enumerate(sched):
        rawbufs = bufs_from_lin(Linearizer(kernel.ast))

        lin = Linearizer(kernel.ast, device.compiler.linearizer_opts)
        # print(get_linearizer_actions(lin))
        # lin.hand_coded_optimizations()
        uops = lin.linearize().uops
        graph_uops(uops)

        tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
        gflops = (
            sym_infer(lin.info.flops, {k: k.min for k in lin.ast.vars()}) * 1e-9 / tm
        )
        total_tm += tm
        running_gflops += gflops * tm
        print(
            f"*** {total_tm*1000:7.2f} ms : kernel{i:2d} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS"
        )
    print(
        f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS"
    )
