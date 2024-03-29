

from functools import partial
import argparse
import numpy as np
import time
import sys

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = 'simple'

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(grpcclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(grpcclient.InferRequestedOutput('OUTPUT0'))
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT1'))

    # Define the callback function. Note the last two parameters should be
    # result and error. InferenceServerClient would povide the results of an
    # inference as grpcclient.InferResult in result. For successful
    # inference, error will be None, otherwise it will be an object of
    # tritonclientutils.InferenceServerException holding the error details
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    # list to hold the results of inference.
    user_data = []

    # Inference call
    triton_client.async_infer(model_name=model_name,
                              inputs=inputs,
                              callback=partial(callback, user_data),
                              outputs=outputs,
                              client_timeout=FLAGS.client_timeout)

    # Wait until the results are available in user_data
    time_out = 10
    while ((len(user_data) == 0) and time_out > 0):
        time_out = time_out - 1
        time.sleep(1)

    # Display and validate the available results
    if ((len(user_data) == 1)):
        # Check for the errors
        if type(user_data[0]) == InferenceServerException:
            print(user_data[0])
            sys.exit(1)

        # Validate the values by matching with already computed expected
        # values.
        output0_data = user_data[0].as_numpy('OUTPUT0')
        output1_data = user_data[0].as_numpy('OUTPUT1')
        for i in range(16):
            print(
                str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
                " = " + str(output0_data[0][i]))
            print(
                str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
                " = " + str(output1_data[0][i]))
            if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
                print("sync infer error: incorrect sum")
                sys.exit(1)
            if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
                print("sync infer error: incorrect difference")
                sys.exit(1)
        print("PASS: Async infer")