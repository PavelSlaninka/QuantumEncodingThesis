# ==============================================================================
# Imports:
# ==============================================================================
from qsp.base import QuantumStatePreparation


# ==============================================================================
# Class:
# ==============================================================================
class BasisEncoding(QuantumStatePreparation):
    """
    The class 'BasisEncoding' implements the basis encoding method for
    encoding a binary input vector into a quantum state.

    The basis encoding method encodes the elements of the input vector
    into qubits using the gates 'x', where '1' values are encoded into
    the quantum state.

    Parameters:
    ----------
    input_vector: list[numbers.Integral]
        The input vector to be encoded into a quantum circuit.
        The elements of the input vector must be integers '0' or '1'
        (raises an error if it is not the case).

    Properties:
    -----------
    input_vector: list[numbers.Integral]
        The vector that is encoded in the quantum circuit.
    qubit_count: int
        The number of qubits used in the quantum circuit.
    quantum_circuit: qiskit.QuantumCircuit
        The quantum circuit that encodes the input vector.
    measured_qubits: list[int]
        A list of the measured qubits.

    Methods:
    --------
    measure() -> None
        Applies the required measurements to the circuit.
    run_aer_simulator(shots, show_plot) -> qiskit.result.result.Result
        Runs the Aer simulator to obtain the encoding results.
    get_statevector() -> qiskit.quantum_info.states.statevector.Statevector
        Gets the statevector of the quantum circuit.

    Examples:
    ---------
    >>> from qsp import BasisEncoding
    >>> basis_encoding = qsp.BasisEncoding([0, 1])
    >>> basis_encoding.get_statevector()
    Statevector([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j], dims=(2, 2))

    Note:
    -----
    This class inherits from the base class 'QuantumStatePreparation',
    which contains code shared by classes implementing
    the other encoding methods.

    This encoding method is explained in detail in the thesis text.
    """

    # ==========================================================================
    # Implemented abstract methods:
    # ==========================================================================
    def _validate_input_vector(self) -> None:
        """
        Validates the input vector binarity required by
        the basis encoding method (checks if the vector
        only contains integers '0' or '1' and, if not,
        raises an error). Then, it calculates the number
        of qubits required to encode the input vector and
        creates a list of the qubits that should
        be measured to obtain the encoding results.

        Raises:
        -------
        ValueError
            If the input vector contains something
            other than integers '0' or '1'.

        Returns:
        --------
        None
        """

        # Iterate over the elements of the input vector:
        for element in self._input_vector:
            # If the element is not an integer
            # '0' or '1', raise an error:
            if element not in (0, 1):
                raise ValueError(
                    "The input vector must only contain integers"
                    " '0' or '1' to use the basis encoding."
                )

        # Calculate the number of qubits required to encode
        # the input vector (in the basis encoding method, it
        # is the same as the length of the input vector):
        self._qubit_count = len(self._input_vector)

        # Create a list of the qubits that should be measured
        # to obtain the encoding results (in the basis encoding
        # method, every qubit should be measured):
        self._measured_qubits = [i for i in range(self._qubit_count)]

    # --------------------------------------------------------------------------
    def _encode_input_vector(self) -> None:
        """
        Encodes the input vector into the quantum
        circuit using the basis encoding method.

        The method iterates over the elements of the input vector and
        applies the gate 'x' to the corresponding qubit if the element
        equals '1'. This encodes the '1' values into the quantum state.

        Returns:
        --------
        None
        """

        # Iterate over the elements of the input vector:
        for index, element in enumerate(self._input_vector):
            # Apply the gate 'x' to the corresponding
            # qubit if the element equals integer '1':
            if element == 1:
                self._quantum_circuit.x(index)
