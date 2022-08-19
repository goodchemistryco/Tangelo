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

"""Simulator class, wrapping around the various simulators and abstracting their
differences from the user. Able to run noiseless and noisy simulations,
leveraging the capabilities of different backends, quantum or classical.

If the user provides a noise model, then a noisy simulation is run with n_shots
shots. If the user only provides n_shots, a noiseless simulation is run, drawing
the desired amount of shots. If the target backend has access to the statevector
representing the quantum state, we leverage any kind of emulation available to
reduce runtime (directly generating shot values from final statevector etc) If
the quantum circuit contains a MEASURE instruction, it is assumed to simulate a
mixed-state and the simulation will be carried by simulating individual shots
(e.g a number of shots is required).

Some backends may only support a subset of the above. This information is
contained in a separate data-structure.
"""

import os
import math
from collections import Counter

import numpy as np
from scipy import stats
from bitarray import bitarray
from openfermion.ops import QubitOperator

from tangelo.helpers.utils import default_simulator
from tangelo.linq import Gate, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
import tangelo.linq.translator as translator


# Data-structure showing what functionalities are supported by the backend, in this package
backend_info = dict()
backend_info["qiskit"] = {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
backend_info["qulacs"] = {"statevector_available": True, "statevector_order": "msq_first", "noisy_simulation": True}
backend_info["cirq"] = {"statevector_available": True, "statevector_order": "lsq_first", "noisy_simulation": True}
backend_info["qdk"] = {"statevector_available": False, "statevector_order": None, "noisy_simulation": False}
backend_info["qiskit_device"] = {"statevector_available": False, "statevector_order": "msq_first", "noisy_simulation": True}


class Simulator:

    def __init__(self, target=default_simulator, n_shots=None, noise_model=None):
        """Instantiate Simulator object.

        Args:
            target (str): One of the available target backends (quantum or
                classical). The default simulator is qulacs if found in user's
                environment, otherwise cirq.
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format
                expected from the target backend.
        """
        self._source = "abstract"
        self._target = target if target else default_simulator
        self._current_state = None
        self._noise_model = noise_model

        # Can be modified later by user as long as long as it retains the same type (ex: cannot change to/from None)
        self.n_shots = n_shots
        self.freq_threshold = 1e-10

        # Set additional attributes related to the target backend chosen by the user
        for k, v in backend_info[self._target].items():
            setattr(self, k, v)

        # Raise error if user attempts to pass a noise model to a backend not supporting noisy simulation
        if self._noise_model and not self.noisy_simulation:
            raise ValueError("Target backend does not support noise models.")

        # Raise error if the number of shots has not been passed for a noisy simulation or if statevector unavailable
        if not self.n_shots and (not self.statevector_available or self._noise_model):
            raise ValueError("A number of shots needs to be specified.")

    def simulate(self, source_circuit, return_statevector=False, initial_statevector=None, qubits_to_use=None, opt_level=1, meas_mitt=False):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit: a circuit in the abstract format to be translated
                for the target backend.
            return_statevector(bool): option to return the statevector as well,
                if available.
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend
            qubits_to_use (list): List of qubits to use for simulation. Must have
                same length as number of qubits in the circuit.
            opt_level (int): optimization level 0,1,2,3
            meas_mitt (bool): use measurement_error_mitigation.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """

        if source_circuit.is_mixed_state and not self.n_shots:
            raise ValueError("Circuit contains MEASURE instruction, and is assumed to prepare a mixed state."
                             "Please set the Simulator.n_shots attribute to an appropriate value.")

        if source_circuit.width == 0:
            raise ValueError("Cannot simulate an empty circuit (e.g identity unitary) with unknown number of qubits.")

        # If the unitary is the identity (no gates) and no noise model, no need for simulation:
        # return all-zero state or sample from statevector
        if source_circuit.size == 0 and not self._noise_model:
            if initial_statevector is not None:
                statevector = initial_statevector
                frequencies = self._statevector_to_frequencies(initial_statevector)
            else:
                frequencies = {'0'*source_circuit.width: 1.0}
                statevector = np.zeros(2**source_circuit.width)
                statevector[0] = 1.0
            return (frequencies, statevector) if return_statevector else (frequencies, None)

        if self._target == "qulacs":
            import qulacs

            translated_circuit = translator.translate_qulacs(source_circuit, self._noise_model)

            # Initialize state on GPU if available and desired. Default to CPU otherwise.
            if ('QuantumStateGpu' in dir(qulacs)) and (int(os.getenv("QULACS_USE_GPU", 0)) != 0):
                state = qulacs.QuantumStateGpu(source_circuit.width)
            else:
                state = qulacs.QuantumState(source_circuit.width)
            if initial_statevector is not None:
                state.load(initial_statevector)

            if (source_circuit.is_mixed_state or self._noise_model):
                samples = list()
                for i in range(self.n_shots):
                    translated_circuit.update_quantum_state(state)
                    samples.append(state.sampling(1)[0])
                    if initial_statevector is not None:
                        state.load(initial_statevector)
                    else:
                        state.set_zero_state()
                python_statevector = None
            elif self.n_shots is not None:
                translated_circuit.update_quantum_state(state)
                python_statevector = np.array(state.get_vector()) if return_statevector else None
                samples = state.sampling(self.n_shots)
            else:
                translated_circuit.update_quantum_state(state)
                self._current_state = state
                python_statevector = state.get_vector()
                frequencies = self._statevector_to_frequencies(python_statevector)
                return (frequencies, np.array(python_statevector)) if return_statevector else (frequencies, None)

            frequencies = {self.__int_to_binstr(k, source_circuit.width): v / self.n_shots
                           for k, v in Counter(samples).items()}
            return (frequencies, python_statevector)

        elif self._target == "qiskit":
            import qiskit

            translated_circuit = translator.translate_qiskit(source_circuit)

            # If requested, set initial state
            if initial_statevector is not None:
                if self._noise_model:
                    raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")
                else:
                    n_qubits = int(math.log2(len(initial_statevector)))
                    initial_state_circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
                    initial_state_circuit.initialize(initial_statevector, list(range(n_qubits)))
                    translated_circuit = initial_state_circuit.compose(translated_circuit)

            # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
            if self._noise_model or source_circuit.is_mixed_state:
                from tangelo.linq.noisy_simulation.noise_models import get_qiskit_noise_model

                meas_range = range(source_circuit.width)
                translated_circuit.measure(meas_range, meas_range)
                return_statevector = False
                backend = qiskit.Aer.get_backend("aer_simulator")

                qiskit_noise_model = get_qiskit_noise_model(self._noise_model) if self._noise_model else None
                opt_level = 0 if self._noise_model else None

                job_sim = qiskit.execute(translated_circuit, backend, noise_model=qiskit_noise_model,
                                         shots=self.n_shots, basis_gates=None, optimization_level=opt_level)
                sim_results = job_sim.result()
                frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

            # Noiseless simulation using the statevector simulator otherwise
            else:
                backend = qiskit.Aer.get_backend("aer_simulator", method='statevector')
                translated_circuit = qiskit.transpile(translated_circuit, backend)
                translated_circuit.save_statevector()
                sim_results = backend.run(translated_circuit).result()
                self._current_state = sim_results.get_statevector(translated_circuit)
                frequencies = self._statevector_to_frequencies(self._current_state)

            return (frequencies, np.array(sim_results.get_statevector())) if return_statevector else (frequencies, None)

        elif self._target == "qdk":

            translated_circuit = translator.translate_qsharp(source_circuit)
            with open('tmp_circuit.qs', 'w+') as f_out:
                f_out.write(translated_circuit)

            # Compile, import and call Q# operation to compute frequencies. Only import qsharp module if qdk is running
            # TODO: A try block to catch an exception at compile time, for Q#? Probably as an ImportError.
            import qsharp
            qsharp.reload()
            from MyNamespace import EstimateFrequencies
            frequencies_list = EstimateFrequencies.simulate(nQubits=source_circuit.width, nShots=self.n_shots)
            print("Q# frequency estimation with {0} shots: \n {1}".format(self.n_shots, frequencies_list))

            # Convert Q# output to frequency dictionary, apply threshold
            frequencies = {bin(i).split('b')[-1]: frequencies_list[i] for i, freq in enumerate(frequencies_list)}
            frequencies = {("0"*(source_circuit.width-len(k))+k)[::-1]: v for k, v in frequencies.items()
                           if v > self.freq_threshold}
            return (frequencies, None)

        elif self._target == "qiskit_device":
            import qiskit
            from qiskit.test import mock
            from qiskit.providers.aer import AerSimulator
            from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

            translated_circuit, q, c = translator.translate_qiskit(source_circuit, qubits_to_use, return_registers=True)

            # If requested, set initial state
            if initial_statevector is not None:
                raise ValueError("Cannot load an initial state if using a noise model, with Qiskit")

            # Drawing individual shots with the qasm simulator, for noisy simulation or simulating mixed states
            num_measure_qubits = len(qubits_to_use) if qubits_to_use is not None else source_circuit.width
            translated_circuit.measure(q, c)

            virtual_to_physical = dict()
            qubit_map = qubits_to_use if qubits_to_use is not None else [i for i in range(num_measure_qubits)]
            if len(qubit_map) != num_measure_qubits:
                raise ValueError("number of qubits_to_use must equal number of qubits in circuit")
            for i in range(num_measure_qubits):
                virtual_to_physical[q[i]] = qubit_map[i]

            return_statevector = False
            if self._noise_model is not None and self._noise_model._device_name is not None:
                try:
                    device_to_call = getattr(mock, self._noise_model._device_name)
                except AttributeError:
                    raise ValueError(f"{self._noise_model._device_name} is not one of the Fake Qiskit backends")
            else:
                raise ValueError("_device_name must be included in a noise_model to run a simulated device")
            device_backend = device_to_call()
            backend = AerSimulator.from_backend(device_backend)

            if meas_mitt:
                meas_calibs, state_labels = complete_meas_cal(qr=q, circlabel='mcal')
                t_qc = qiskit.transpile(meas_calibs, backend, initial_layout=virtual_to_physical)
                qobj = qiskit.assemble(t_qc, shots=10000)
                cal_results = backend.run(qobj, shots=10000).result()
                meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
                meas_filter = meas_fitter.filter

            job_sim = qiskit.execute(qiskit.transpile(translated_circuit, backend, initial_layout=virtual_to_physical, optimization_level=opt_level), backend,
                                     shots=self.n_shots, basis_gates=None)
            if meas_mitt:
                presim_results = job_sim.result()
                sim_results = meas_filter.apply(presim_results)
            else:
                sim_results = job_sim.result()

            frequencies = {state[::-1]: count/self.n_shots for state, count in sim_results.get_counts(0).items()}

            return (frequencies, np.array(sim_results.get_statevector())) if return_statevector else (frequencies, None)

        elif self._target == "qdk":

            translated_circuit = translator.translate_qsharp(source_circuit)
            with open('tmp_circuit.qs', 'w+') as f_out:
                f_out.write(translated_circuit)

            # Compile, import and call Q# operation to compute frequencies. Only import qsharp module if qdk is running
            # TODO: A try block to catch an exception at compile time, for Q#? Probably as an ImportError.
            import qsharp
            qsharp.reload()
            from MyNamespace import EstimateFrequencies
            frequencies_list = EstimateFrequencies.simulate(nQubits=source_circuit.width, nShots=self.n_shots)
            print("Q# frequency estimation with {0} shots: \n {1}".format(self.n_shots, frequencies_list))

            # Convert Q# output to frequency dictionary, apply threshold
            frequencies = {bin(i).split('b')[-1]: frequencies_list[i] for i, freq in enumerate(frequencies_list)}
            frequencies = {("0"*(source_circuit.width-len(k))+k)[::-1]: v for k, v in frequencies.items()
                           if v > self.freq_threshold}
            return (frequencies, None)

        elif self._target == "cirq":
            import cirq

            translated_circuit = translator.translate_cirq(source_circuit, self._noise_model)

            if source_circuit.is_mixed_state or self._noise_model:
                # Only DensityMatrixSimulator handles noise well, can use Simulator but it is slower
                cirq_simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
            else:
                cirq_simulator = cirq.Simulator(dtype=np.complex128)

            # If requested, set initial state
            cirq_initial_statevector = initial_statevector if initial_statevector is not None else 0

            # Calculate final density matrix and sample from that for noisy simulation or simulating mixed states
            if self._noise_model or source_circuit.is_mixed_state:
                # cirq.dephase_measurements changes measurement gates to Krauss operators so simulators
                # can be called once and density matrix sampled repeatedly.
                translated_circuit = cirq.dephase_measurements(translated_circuit)
                sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                self._current_state = sim.final_density_matrix
                indices = list(range(source_circuit.width))
                isamples = cirq.sample_density_matrix(sim.final_density_matrix, indices, repetitions=self.n_shots)
                samples = [''.join([str(int(q))for q in isamples[i]]) for i in range(self.n_shots)]

                frequencies = {k: v / self.n_shots for k, v in Counter(samples).items()}
            # Noiseless simulation using the statevector simulator otherwise
            else:
                job_sim = cirq_simulator.simulate(translated_circuit, initial_state=cirq_initial_statevector)
                self._current_state = job_sim.final_state_vector
                frequencies = self._statevector_to_frequencies(self._current_state)

            return (frequencies, np.array(self._current_state)) if return_statevector else (frequencies, None)

    def get_expectation_value(self, qubit_operator, state_prep_circuit, initial_statevector=None, qubits_to_use=None, opt_level=1, meas_mitt=False):
        r"""Take as input a qubit operator H and a quantum circuit preparing a
        state |\psi>. Return the expectation value <\psi | H | \psi>.

        In the case of a noiseless simulation, if the target backend exposes the
        statevector then it is used directly to compute expectation values, or
        draw samples if required. In the case of a noisy simulator, or if the
        statevector is not available on the target backend, individual shots
        must be run and the workflow is akin to what we would expect from an
        actual QPU.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit: an abstract circuit used for state preparation.
            qubits_to_use (list): list of physical qubits to use
            opt_level (int): optimation level 0,1,2,3
            meas_mitt (bool): Whether to use measurement_error_mitigation

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        # Check if simulator supports statevector
        if initial_statevector is not None and not self.statevector_available:
            raise ValueError(f'Statevector not supported in {self._target}')

        # Check that qubit operator does not operate on qubits beyond circuit size.
        # Keep track if coefficients are real or not
        are_coefficients_real = True
        for term, coef in qubit_operator.terms.items():
            if state_prep_circuit.width < len(term):
                raise ValueError(f'Term {term} requires more qubits than the circuit contains ({state_prep_circuit.width})')
            if type(coef) in {complex, np.complex64, np.complex128}:
                are_coefficients_real = False

        # If the underlying operator is hermitian, expectation value is real and can be computed right away
        if are_coefficients_real:
            if self._noise_model or not self.statevector_available \
                    or state_prep_circuit.is_mixed_state or state_prep_circuit.size == 0:
                return self._get_expectation_value_from_frequencies(qubit_operator, state_prep_circuit, initial_statevector=initial_statevector,
                                                                    qubits_to_use=qubits_to_use, opt_level=opt_level, meas_mitt=meas_mitt)
            elif self.statevector_available:
                return self._get_expectation_value_from_statevector(qubit_operator, state_prep_circuit, initial_statevector=initial_statevector)

        # Else, separate the operator into 2 hermitian operators, use linearity and call this function twice
        else:
            qb_op_real, qb_op_imag = QubitOperator(), QubitOperator()
            for term, coef in qubit_operator.terms.items():
                qb_op_real.terms[term], qb_op_imag.terms[term] = coef.real, coef.imag
            qb_op_real.compress()
            qb_op_imag.compress()
            exp_real = self.get_expectation_value(qb_op_real, state_prep_circuit, initial_statevector=initial_statevector,
                                                  qubits_to_use=qubits_to_use, opt_level=opt_level, meas_mitt=meas_mitt)
            exp_imag = self.get_expectation_value(qb_op_imag, state_prep_circuit, initial_statevector=initial_statevector,
                                                  qubits_to_use=qubits_to_use, opt_level=opt_level, meas_mitt=meas_mitt)
            return exp_real if (exp_imag == 0.) else exp_real + 1.0j * exp_imag

    def _get_expectation_value_from_statevector(self, qubit_operator, state_prep_circuit, initial_statevector=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi>, computed
        without drawing samples (statevector only). Users should not be calling
        this function directly, please call "get_expectation_value" instead.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit: an abstract circuit used for state preparation
                (only pure states).

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width

        expectation_value = 0.
        prepared_frequencies, prepared_state = self.simulate(state_prep_circuit, return_statevector=True, initial_statevector=initial_statevector)

        # Use fast built-in qulacs expectation value function if possible
        if self._target == "qulacs" and not self.n_shots:
            import qulacs

            # Note: This section previously used qulacs.quantum_operator.create_quantum_operator_from_openfermion_text but was changed
            # due to a memory leak. We can re-evaluate the implementation if/when Issue #303 (https://github.com/qulacs/qulacs/issues/303)
            # is fixed.
            operator = qulacs.Observable(n_qubits)
            for term, coef in qubit_operator.terms.items():
                pauli_string = "".join(f" {op} {qu}" for qu, op in term)
                operator.add_operator(coef, pauli_string)
            return operator.get_expectation_value(self._current_state).real

        # Use cirq built-in expectation_from_state_vector/epectation_from_density_matrix
        # noise model would require
        if self._target == "cirq" and not self.n_shots:
            import cirq

            GATE_CIRQ = translator.get_cirq_gates()
            qubit_labels = cirq.LineQubit.range(n_qubits)
            qubit_map = {q: i for i, q in enumerate(qubit_labels)}
            paulisum = 0.*cirq.PauliString(cirq.I(qubit_labels[0]))
            for term, coef in qubit_operator.terms.items():
                pauli_list = [GATE_CIRQ[pauli](qubit_labels[index]) for index, pauli in term]
                paulisum += cirq.PauliString(pauli_list, coefficient=coef)
            if self._noise_model:
                exp_value = paulisum.expectation_from_density_matrix(prepared_state, qubit_map)
            else:
                exp_value = paulisum.expectation_from_state_vector(prepared_state, qubit_map)
            return np.real(exp_value)

        # Otherwise, use generic statevector expectation value
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:  # Cannot have a qubit index beyond circuit size
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            if not self.n_shots:
                # Directly simulate and compute expectation value using statevector
                pauli_circuit = Circuit([Gate(pauli, index) for index, pauli in term], n_qubits=n_qubits)
                _, pauli_state = self.simulate(pauli_circuit, return_statevector=True, initial_statevector=prepared_state)

                delta = np.dot(pauli_state.real, prepared_state.real) + np.dot(pauli_state.imag, prepared_state.imag)
                expectation_value += coef * delta

            else:
                # Run simulation with statevector but compute expectation value with samples directly drawn from it
                basis_circuit = Circuit(measurement_basis_gates(term), n_qubits=state_prep_circuit.width)
                if basis_circuit.size > 0:
                    frequencies, _ = self.simulate(basis_circuit, initial_statevector=prepared_state)
                else:
                    frequencies = prepared_frequencies
                expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
                expectation_value += coef * expectation_term

        return expectation_value

    def _get_expectation_value_from_frequencies(self, qubit_operator, state_prep_circuit, initial_statevector=None,
                                                qubits_to_use=None, opt_level=1, meas_mitt=False):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi> computed
        using the frequencies of observable states.

        Args:
            qubit_operator(openfermion-style QubitOperator class): a qubit
                operator.
            state_prep_circuit: an abstract circuit used for state preparation.
            qubits_to_use (list): list of physical qubits to use on device
            opt_level (int): Optimization level for compiling 0,1,2,3
            meas_mitt (bool): Whether to use measurement_error_mitigation

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width
        if not self.statevector_available or state_prep_circuit.is_mixed_state or self._noise_model:
            initial_circuit = state_prep_circuit
            if initial_statevector is not None and not self.statevector_available:
                raise ValueError(f'Backend {self._target} does not support statevectors')
            else:
                updated_statevector = initial_statevector
        else:
            initial_circuit = Circuit(n_qubits=n_qubits)
            _, updated_statevector = self.simulate(state_prep_circuit, return_statevector=True, initial_statevector=initial_statevector,
                                                   qubits_to_use=qubits_to_use, opt_level=opt_level, meas_mitt=meas_mitt)

        expectation_value = 0.
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            basis_circuit = Circuit(measurement_basis_gates(term))
            full_circuit = initial_circuit + basis_circuit if (basis_circuit.size > 0) else initial_circuit
            frequencies, _ = self.simulate(full_circuit, initial_statevector=updated_statevector,
                                           qubits_to_use=qubits_to_use, opt_level=opt_level, meas_mitt=meas_mitt)
            expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
            expectation_value += coef * expectation_term

        return expectation_value

    @staticmethod
    def get_expectation_value_from_frequencies_oneterm(term, frequencies):
        """Return the expectation value of a single-term qubit-operator, given
        the result of a state-preparation.

        Args:
            term(openfermion-style QubitOperator object): a qubit operator, with
                only a single term.
            frequencies(dict): histogram of frequencies of measurements (assumed
                to be in lsq-first format).

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """

        if not frequencies.keys():
            return ValueError("Must pass a non-empty dictionary of frequencies.")
        n_qubits = len(list(frequencies.keys())[0])

        # Get term mask
        mask = ["0"] * n_qubits
        for index, op in term:
            mask[index] = "1"
        mask = "".join(mask)

        # Compute expectation value of the term
        expectation_term = 0.
        for basis_state, freq in frequencies.items():
            # Compute sample value using state_binstr and term mask, update term expectation value
            sample = (-1) ** ((bitarray(mask) & bitarray(basis_state)).to01().count("1") % 2)
            expectation_term += sample * freq

        return expectation_term

    def _statevector_to_frequencies(self, statevector):
        """For a given statevector representing the quantum state of a qubit
        register, returns a sparse histogram of the probabilities in the
        least-significant-qubit (lsq) -first order. e.g the string '100' means
        qubit 0 measured in basis state |1>, and qubit 1 & 2 both measured in
        state |0>.

        Args:
            statevector(list or ndarray(complex)): an iterable 1D data-structure
                containing the amplitudes.

        Returns:
            dict: A dictionary whose keys are bitstrings representing the
                multi-qubit states with the least significant qubit first (e.g
                '100' means qubit 0 in state |1>, and qubit 1 and 2 in state
                |0>), and the associated value is the corresponding frequency.
                Unless threshold=0., this dictionary will be sparse.
        """

        n_qubits = int(math.log2(len(statevector)))
        frequencies = dict()
        for i, amplitude in enumerate(statevector):
            frequency = abs(amplitude)**2
            if (frequency - self.freq_threshold) >= 0.:
                frequencies[self.__int_to_binstr(i, n_qubits)] = frequency

        # If n_shots, has been specified, then draw that amount of samples from the distribution
        # and return empirical frequencies instead. Otherwise, return the exact frequencies
        if not self.n_shots:
            return frequencies
        else:
            xk, pk = [], []
            for k, v in frequencies.items():
                xk.append(int(k[::-1], 2))
                pk.append(frequencies[k])
            distr = stats.rv_discrete(name='distr', values=(np.array(xk), np.array(pk)))

            # Generate samples from distribution. Cut in chunks to ensure samples fit in memory, gradually accumulate
            chunk_size = 10**7
            n_chunks = self.n_shots // chunk_size
            freqs_shots = Counter()

            for i in range(n_chunks+1):
                this_chunk = self.n_shots % chunk_size if i == n_chunks else chunk_size
                samples = distr.rvs(size=this_chunk)
                freqs_shots += Counter(samples)
            freqs_shots = {self.__int_to_binstr_lsq(k, n_qubits): v / self.n_shots for k, v in freqs_shots.items()}
            return freqs_shots

    def __int_to_binstr(self, i, n_qubits):
        """Convert an integer into a bit string of size n_qubits, in the order
        specified for the state vector.
        """
        bs = bin(i).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs
        return state_binstr[::-1] if (self.statevector_order == "msq_first") else state_binstr

    def __int_to_binstr_lsq(self, i, n_qubits):
        """Convert an integer into a bit string of size n_qubits, in the
        least-significant qubit order.
        """
        bs = bin(i).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs
        return state_binstr[::-1]
