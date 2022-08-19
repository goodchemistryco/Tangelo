# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic noise model representation and backend-specific translation function.
The Simulator class is responsible for taking in the generic noise model at
runtime and applying it accordingly to the target compute backend.
Only works for simulators supporting noisy simulation.
"""


from tangelo.linq import ONE_QUBIT_GATES

SUPPORTED_NOISE_MODELS = {'depol', 'pauli'}

__MAPPING_GATES_QISKIT = dict()
for name in ONE_QUBIT_GATES:
    __MAPPING_GATES_QISKIT[name] = [name.lower()]
for name in {'RX', 'RY', 'RZ'}:
    __MAPPING_GATES_QISKIT[name] = ['u1', 'u2', 'u3']
__MAPPING_GATES_QISKIT["CNOT"] = ["cx"]


class NoiseModel:
    """A class representing a noise model. Not specific to any compute backend.
    The object holds a dictionary mapping each noisy gate to a list of noise
    that should be applied every time it is encountered in a circuit.

    Pauli noise expects a list of 3 probabilities corresponding to X,Y,Z pauli
    noises (I is deduced from the sum) Depolarization noise is a special case of
    Pauli noise, where the 3 probabilities are equal: a float is enough. Please
    check the notebook tutorial in the example section more a more in-depth
    description of the different noise channels.
    """

    def __init__(self, device_name=None):
        self._quantum_errors = dict()
        self._device_name = device_name

    def __repr__(self):
        return str(self._quantum_errors)

    def add_quantum_error(self, abs_gate, noise_type, noise_params):
        """Adds the desired noise to the gate specified by user. Checks if
        inputs makes sense.
        """

        if noise_type not in SUPPORTED_NOISE_MODELS:
            raise ValueError(f"Error model not supported :{noise_type}\nCurrently supported:{SUPPORTED_NOISE_MODELS}")
        if noise_type == 'pauli' and (not isinstance(noise_params, list) or len(noise_params) != 3):
            raise ValueError(f"For pauli noise, a list of 3 probabilities (corresponding to X,Y,Z) is expected")
        if noise_type == 'depol' and not isinstance(noise_params, float):
            raise ValueError(f"For depolarization noise, the expected parameter must be a single float")

        if abs_gate in self._quantum_errors:
            if noise_type not in {nt for nt, np in self._quantum_errors[abs_gate]}:
                self._quantum_errors[abs_gate] += [(noise_type, noise_params)]
            else:
                raise ValueError(f'Cannot cumulate several noise channels of the same type on the same gate')
        else:
            self._quantum_errors[abs_gate] = [(noise_type, noise_params)]

    @property
    def noisy_gates(self):
        return set(self._quantum_errors.keys())


def get_qiskit_noise_dict(noise_model):
    """Takes in generic noise model, converts to a dictionary whose keys are
    the qiskit basis gates supported by the qasm simulator, to the noises,
    ensuring there are no redundancy / duplicates on the U-gates in particular.
    """

    qnd = dict()
    for gate, noises in noise_model._quantum_errors.items():
        for qiskit_gate in __MAPPING_GATES_QISKIT[gate]:
            if qiskit_gate not in qnd:
                qnd[qiskit_gate] = noises
            else:
                noise_types = [nt for nt, np in qnd[qiskit_gate]]
                for noise in noises:
                    if noise[0] not in noise_types:
                        noise_types.append(noise[0])
                        qnd[qiskit_gate].append(noise)
    return qnd


def get_qiskit_noise_model(noise_model):
    """Takes a NoiseModel object as input, returns an equivalent Qiskit Noise
    Model, compatible with QASM simulator.
    """

    from qiskit.providers.aer.noise import NoiseModel as QiskitNoiseModel
    from qiskit.providers.aer.noise.errors import depolarizing_error, pauli_error

    qnd = get_qiskit_noise_dict(noise_model)
    qnm = QiskitNoiseModel()
    for gate, noises in qnd.items():
        for noise, probs in noises:
            if noise == 'pauli':
                qerr1 = pauli_error(list(zip(['X', 'Y', 'Z'], probs)) + [('I', 1-sum(probs))])
                qerr2 = qerr1.tensor(qerr1)
                qerr = qerr1 if gate in qnm._1qubit_instructions else qerr2
            elif noise == 'depol':
                qerr = depolarizing_error(probs, 1 if gate in qnm._1qubit_instructions else 2)
            else:
                raise ValueError(f"Error model currently not supported with Qiskit backend :{noise}")
            qnm.add_all_qubit_quantum_error(qerr, gate)
    return qnm
