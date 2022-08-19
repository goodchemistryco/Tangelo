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

"""Functions helping with quantum circuit format conversion between abstract
format and qiskit format.

In order to produce an equivalent circuit for the target backend, it is
necessary to account for:
- how the gate names differ between the source backend to the target backend.
- how the order and conventions for some of the inputs to the gate operations
    may also differ.
"""


def get_qiskit_gates():
    """Map gate name of the abstract format to the equivalent add_gate method of
    Qiskit's QuantumCircuit class API and supported gates:
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
    """

    import qiskit

    GATE_QISKIT = dict()
    GATE_QISKIT["H"] = qiskit.QuantumCircuit.h
    GATE_QISKIT["X"] = qiskit.QuantumCircuit.x
    GATE_QISKIT["Y"] = qiskit.QuantumCircuit.y
    GATE_QISKIT["Z"] = qiskit.QuantumCircuit.z
    GATE_QISKIT["CH"] = qiskit.QuantumCircuit.ch
    GATE_QISKIT["CX"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["CY"] = qiskit.QuantumCircuit.cy
    GATE_QISKIT["CZ"] = qiskit.QuantumCircuit.cz
    GATE_QISKIT["S"] = qiskit.QuantumCircuit.s
    GATE_QISKIT["T"] = qiskit.QuantumCircuit.t
    GATE_QISKIT["RX"] = qiskit.QuantumCircuit.rx
    GATE_QISKIT["RY"] = qiskit.QuantumCircuit.ry
    GATE_QISKIT["RZ"] = qiskit.QuantumCircuit.rz
    GATE_QISKIT["CRX"] = qiskit.QuantumCircuit.crx
    GATE_QISKIT["CRY"] = qiskit.QuantumCircuit.cry
    GATE_QISKIT["CRZ"] = qiskit.QuantumCircuit.crz
    GATE_QISKIT["CNOT"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["SWAP"] = qiskit.QuantumCircuit.swap
    GATE_QISKIT["XX"] = qiskit.QuantumCircuit.rxx
    GATE_QISKIT["CSWAP"] = qiskit.QuantumCircuit.cswap
    GATE_QISKIT["PHASE"] = qiskit.QuantumCircuit.p
    GATE_QISKIT["CPHASE"] = qiskit.QuantumCircuit.cp
    GATE_QISKIT["MEASURE"] = qiskit.QuantumCircuit.measure
    return GATE_QISKIT


def translate_qiskit(source_circuit, qubits_to_use=None, return_registers=False):
    """Take in an abstract circuit, return an equivalent qiskit QuantumCircuit
    instance

    Args:
        source_circuit: quantum circuit in the abstract format.
        qubits_to_use: list
        return_registers (bool): whether to return the registers to simulate

    Returns:
        qiskit.QuantumCircuit: the corresponding qiskit quantum circuit. if return_registers=False
        (qiskit.QuantumCircuit, qiskit.QuantumRegister, qiskit.ClassicalRegister) if return_registers=True
    """

    import qiskit

    GATE_QISKIT = get_qiskit_gates()
    num_virtual_qubits = len(source_circuit._qubit_indices) if qubits_to_use is not None else source_circuit.width
    q = qiskit.QuantumRegister(num_virtual_qubits, name="q")
    c = qiskit.ClassicalRegister(num_virtual_qubits, name="c")
    target_circuit = qiskit.QuantumCircuit(q, c)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.control is not None:
            if len(gate.control) > 1:
                raise ValueError('Multi-controlled gates not supported with qiskit. Gate {gate.name} with controls {gate.control} is not allowed')
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]])
        elif gate.name in {"RX", "RY", "RZ", "PHASE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.target[0]])
        elif gate.name in {"CRX", "CRY", "CRZ", "CPHASE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.control[0]], q[gate.target[0]])
        elif gate.name in {"CNOT", "CH", "CX", "CY", "CZ"}:
            (GATE_QISKIT[gate.name])(target_circuit, q[gate.control[0]], q[gate.target[0]])
        elif gate.name in {"SWAP"}:
            (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]], q[gate.target[1]])
        elif gate.name in {"CSWAP"}:
            (GATE_QISKIT[gate.name])(target_circuit, q[gate.control[0]], q[gate.target[0]], q[gate.target[1]])
        elif gate.name in {"XX"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, q[gate.target[0]], q[gate.target[1]])
        elif gate.name in {"MEASURE"}:
            (GATE_QISKIT[gate.name])(target_circuit, q[gate.target[0]], c[gate.target[0]])
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qiskit")
    if return_registers:
        return target_circuit, q, c
    else:
        return target_circuit
