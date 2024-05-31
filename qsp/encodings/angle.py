# ==============================================================================
# Imports:
# ==============================================================================
from qsp.base import QuantumStatePreparation
import numpy as np


# ==============================================================================
# Class:
# ==============================================================================
class AngleEncoding(QuantumStatePreparation):
    """
    The class 'AngleEncoding' implements the angle encoding method
    for encoding a normalized input vector into a quantum state.

    The angle encoding method encodes the elements of the input vector
    into qubits using the gates 'ry'. The encoding process is outlined
    in more detail in the docstring of the _encode_input_vector() method.

    Parameters:
    ----------
    input_vector: list[numbers.Real]
        The input vector to be encoded into a quantum circuit.
        It is normalized if its norm is not equal to one.

    Properties:
    -----------
    input_vector: list[numbers.Real]
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
    >>> from qsp import AngleEncoding
    >>> angle_encoding = qsp.AngleEncoding([-2, 3])
    >>> angle_encoding.get_statevector()
    The input vector with the norm of 3.605551275463989 has been normalized.
    Statevector([0.46153846+0.j, -0.30769231+0.j,  0.69230769+0.j, -0.46153846+0.j], dims=(2, 2))

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
        Calculates the number of qubits required to encode
        the input vector and creates a list of the qubits that
        should be measured to obtain the encoding results. Also
        normalizes the input vector if its norm is not equal to one.

        Returns:
        --------
        None
        """

        # Calculate the number of qubits required to encode
        # the input vector (in the angle encoding method, it
        # is the same as the length of the input vector):
        self._qubit_count = len(self._input_vector)

        # Create a list of the qubits that should be measured
        # to obtain the encoding results (in the angle encoding
        # method, every qubit should be measured):
        self._measured_qubits = [i for i in range(self._qubit_count)]

        # Normalize the input vector if its norm is not
        # equal to one (it is important in the angle encoding
        # method for the input vector to be normalized):
        super()._normalize_input_vector()

    # --------------------------------------------------------------------------
    def _encode_input_vector(self) -> None:
        """
        Encodes the input vector into the quantum
        circuit using the angle encoding method.

        This method embeds the input vector into
        the quantum circuit using the gates 'ry'.

        The rotational gate 'ry' is applied to each qubit. The angle
        of each rotation is given by the inverse of the arcsine
        of the corresponding element in the input vector, multiplied
        by two. This is because the gate rotation rotates the
        state |0> to the state (cos(theta/2), sin(theta/2)), where
        the sin(theta/2) must preferably be equal to the corresponding
        input vector element). This way, the gates 'ry' encode the
        values of the input vector directly into the quantum state
        so that each element of the input vector is equal to:
        - when squared, the probability of the corresponding
          qubit collapsing to the state |1>
        - the second element of the statevector
          of the corresponding qubit

        Returns:
        --------
        None
        """

        # Iterate over the elements of the input vector:
        for index, element in enumerate(self._input_vector):
            # The 'element' is encoded to the circuit using the gate
            # 'ry' below. The right angle must be calculated for the
            # encoding to work correctly (the gate 'ry' produces
            # (cos(theta/2), sin(theta/2)) state, so the inverse
            # must be calculated for the resulting amplitude to
            # be equal to the 'element'):
            angle = 2 * np.arcsin(element)

            # Apply the gate 'ry' to the right qubit:
            self._quantum_circuit.ry(angle, index)
