# ==============================================================================
# Imports:
# ==============================================================================
from qsp.base import QuantumStatePreparation
from qsp import AmplitudeEncoding
import numpy as np
import math
from qiskit.quantum_info import partial_trace, DensityMatrix
import qiskit
from IPython.display import display


# ==============================================================================
# Class:
# ==============================================================================
class DivideAndConquerEncoding(QuantumStatePreparation):
    """
    The class 'DivideAndConquerEncoding' implements the
    divide-and-conquer encoding method for encoding a
    normalized input vector into a quantum state. The
    divide-and-conquer encoding method uses a modified
    version of the amplitude encoding method, where the
    ordinary gates 'ry' are used along with controlled
    gates 'swap' and ancilla qubits. The gates 'cswap'
    are applied in a bottom-up binary tree-like fashion.

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
    >>> from qsp import DivideAndConquerEncoding
    >>> import numpy as np
    >>> input_vector = [0.47286624,  0.63048832, 0.3940552, 0.47286624]
    >>> print("Expected probabilities:", np.power(input_vector, 2))
    >>> encoding = qsp.DivideAndConquerEncoding(input_vector)
    >>> print("Resulting probabilities:", encoding.get_probabilities())
    Expected probabilities: [0.22360248 0.39751552 0.1552795  0.22360248]
    Resulting probabilities: [0.22360248 0.39751552 0.15527951 0.22360249]

    Note:
    -----
    This class inherits from the class 'AmplitudeEncoding', which
    contains the angle calculation method required by this encoding method.

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

        # Get the length of the input vector:
        length = len(self._input_vector)

        # Typically in this encoding method, the input vector's length
        # equals a power of two 'n=2^m'. In such a case, the number of
        # qubits required to encode the input vector equals 'n-1'.
        # However, if 'n!=2^m', the number of qubits must still equal
        # a power of two minus one. So, a power of two larger than the
        # input vector's length is chosen in those cases. All cases
        # are encompassed in the following formula:
        self._qubit_count = 2 ** math.ceil(math.log2(length)) - 1

        # In this encoding method, the gates 'cswap' are used,
        # where each uses 3 qubits. At least one gate 'cswap'
        # is required, so at least 3 qubits are required to
        # encode an input vector of any given size:
        if self._qubit_count < 3:
            # If the qubit count was calculated to
            # be less than 3, set it manually to 3:
            self._qubit_count = 3

        # In this encoding method, not all qubits should be measured
        # to obtain the encoding results. Only those qubits whose
        # indexes equal a power of two are measured (so for example,
        # qubits with the indexes 0, 1, 2, 4, 8, etc...):
        self._measured_qubits = [
            2**i - 1 for i in range(int(math.log2(self._qubit_count + 1)))
        ]

        # Normalize the input vector if its norm is not equal
        # to one (it is important in the divide-and-conquer
        # encoding method for the input vector to be normalized):
        super()._normalize_input_vector()

    # --------------------------------------------------------------------------
    def _encode_input_vector(self) -> None:
        """
        Encodes the input vector using the divide-and-conquer
        amplitude encoding method. The input vector elements
        are encoded into the quantum state using the 'ry'
        gates and the 'swap' gates. One layer of the 'ry'
        gates is used and then the controlled 'swap' gates
        are applied based on a binary-tree traversal method.

        Returns:
        --------
        None
        """

        # Calculate the angles that need to be used in the
        # gates 'ry' (the method 'calculate_amplitude_angles'
        # from the parent class 'AmplitudeEncoding' needs to be
        # called; the number of angles calculated is the same
        # as the number of qubits this encoding method uses):
        angles = AmplitudeEncoding.calculate_amplitude_angles(
            self.input_vector, int(np.log2(self._qubit_count + 1))
        )

        # The divide-and-conquer encoding method works by first applying
        # one gate 'ry' to each qubit. The corresponding angle is used in
        # each rotation (by iterating over the qubit indices and the angles):
        for index, angle in enumerate(angles):
            self._quantum_circuit.ry(angle, index)

        # After the gates 'ry' are applied, the encoded information needs to
        # become concentrated in those qubits that are measured (when obtaining
        # the encoding results in this encoding method, only those qubits need
        # to be measured whose indexes equal a power of two). This means that the
        # encoded information from the unmeasured qubits must somehow become
        # represented in the measured qubits. This can only be achieved using
        # multi-qubit gates. Using them, it is possible to entangle qubits. In
        # this encoding method, the gates 'cswap' are used. This gate is crucial
        # for redistributing quantum information from unmeasured qubits to those
        # that are measured in a structured way. This encoding method works by
        # using those gates in a bottom-up binary tree-like fashion, where the
        # controlled swap gates act across different layers of a binary tree
        # structure formed by the qubits. This hierarchical use of the gates
        # 'cswap' ensures that information from a broad set of qubits is
        # effectively concentrated into a smaller subset containing the
        # measured qubits. The qubit entanglement in this case works such
        # that the state of any unmeasured qubit influences the state of the
        # measured qubits, thus preserving the overall quantum information while
        # focusing it where it can be directly observed. The process uses a
        # bottom-up approach, where the lower level qubits (those further
        # from the root, meaning closer the the end of the quantum circuit)
        # have their information pushed upwards towards the root and other
        # high-level nodes that represent powers of two, ensuring that all
        # relevant information is captured during measurement. This strategy is
        # called 'divide-and-conquer', where the original problem is continually
        # divided into smaller subproblems (subsets of qubits) with their
        # solutions being combined (qubit entanglement). Compared to the
        # amplitude encoding method, this encoding method lowers the need
        # for deep circuits and extensive qubit interactions, which are
        # resource-intensive and increase the likelihood of errors.

        # Two simple functions are defined to help with the binary tree traversal:
        def get_left_node(index: int) -> int:
            """
            Calculates and returns the index of the left child of a given node
            in a binary tree, assuming the tree is represented as an array. In
            this representation, the left child of a node located at index
            'index' is found at position '2 * index + 1'.

            Parameters:
            -----------
            index: int
                The index of the current node in the
                binary tree's array representation.

            Returns:
            --------
            int
                The index of the left child node.
            """
            return 2 * index + 1

        def get_right_node(index: int) -> int:
            """
            Calculates and returns the index of the right child of a given node
            in a binary tree, assuming the tree is represented as an array. In
            this representation, the right child of a node located at index
            'index' is found at position '2 * index + 2'.

            Parameters:
            -----------
            index: int
                The index of the current node in the
                binary tree's array representation.

            Returns:
            --------
            int
                The index of the right child node.
            """
            return 2 * index + 2

        # This for-loop below initializes the variable 'current_qubit' from a
        # midpoint in the binary tree structure of qubits and iterates backwards
        # to the root. The choice of starting point and direction facilitates a
        # bottom-up approach. This variable 'current_qubit' serves as a pointer
        # to the control qubit used in each applied controlled swap gate. The qubit
        # count is always '2^n-1' (n>1), meaning the variable 'current_qubit' is
        # initially set to '(2^n-3)//2', which is 0, 2, 6, 14, 30, etc... for
        # qubits counts 3, 7, 15, 31, 63, etc..., respectively. The loop then
        # decrements the index 'current_qubit' by one in each iteration (the
        # parameter 'step' set to -1), stopping at the first qubit
        # ('current_qubit' = 0, the parameter 'stop' set to -1):
        for current_qubit in range((self._qubit_count - 2) // 2, -1, -1):
            # Retrieve the index of the left child for the current
            # qubit (in binary tree representations, the left child
            # of a node at index 'i' is located at '2*i + 1'):
            left_qubit = get_left_node(current_qubit)

            # Similar to the left child, retrieve
            # the index of the right child (at '2*i + 2'):
            right_qubit = get_right_node(current_qubit)

            # The while loop below ensures that the right child qubit index does
            # not exceed the number of qubits available, preventing errors and
            # ensuring that the quantum operations remain within the bounds of
            # the quantum circuit (the right qubit and its children are further
            # down the qubit list than the left qubit and its children). In each
            # loop iteration, the gates 'cswap' are applied, where the control
            # qubit is defined by the index 'current_qubit', and for its control
            # state |1> the qubits defined by the indexes 'left_qubit' and
            # 'right_qubit' are swapped. While the right qubit is not out of
            # bounds, the while loop iterates while setting both left
            # and right qubits to their left children:
            while right_qubit < self._qubit_count:
                # Apply the gate 'cswap':
                self._quantum_circuit.cswap(current_qubit, left_qubit, right_qubit)

                # Update the child qubit indexes to their own respective
                # left children, allowing the algorithm to progress deeper
                # into the binary tree (in the quantum circuit, the number of
                # qubits between the left and right qubits is '2^n-1' in the
                # 'n'-th loop iteration, while the control qubit does not change):
                left_qubit = get_left_node(left_qubit)
                right_qubit = get_left_node(right_qubit)

    # ==========================================================================
    # Private methods:
    # ==========================================================================
    def _reverse_endianness(self, statevector: np.ndarray) -> np.ndarray:
        """
        Reverses the endianness of a given statevector. The endianness
        of a statevector refers to the order in which the qubits are
        represented. In this method, the endianness is reversed by
        swapping the positions of the qubits.

        The method works by first determining the number of qubits
        in the statevector. Then, it defines a helper function
        'reverse_bits' to reverse the binary representation of
        a given index. Next, it creates a new list 'new_indices' by
        applying the 'reverse_bits' function to each index in the
        range of the statevector length. Finally, it creates a new
        statevector 'reversed_statevector' by indexing the original
        statevector with the 'new_indices'.

        Parameters:
        -----------
        statevector: np.ndarray
            The statevector that needs to have its endianness reversed.

        Returns:
        -------
        np.ndarray
            A statevector that has the reversed order of qubits.
        """

        # Get the number of qubits that comprise the given statevector:
        qubit_count = int(np.log2(len(statevector)))

        def reverse_bits(index: int) -> int:
            """
            Reverses the bits of an integer index based on
            a specified bit width (defined by 'qubit_count').

            Parameters:
            -----------
            index: int
                The qubit index whose bits are to be reversed.

            Returns:
            --------
            int
                The integer corresponding to the
                reversed bit string of the original index.
            """

            # Format the index as a binary string
            # padded to the length of 'qubit_count':
            binary = f"{index:0{qubit_count}b}"

            # Reverse the binary string by
            # using the slicing method [::-1]:
            reversed_binary = binary[::-1]

            # Convert the reversed binary string
            # back to an integer and return it:
            return int(reversed_binary, 2)

        # Create a list by applying 'reverse_bits' on each qubit
        # index, thus creating a permutation of the qubit indices:
        new_indices = [reverse_bits(i) for i in range(len(statevector))]

        # Create a permutation of the original statevector based
        # on 'new_indices' (this will yield the desired
        # statevector that has reversed endianness):
        reversed_statevector = statevector[new_indices]

        return reversed_statevector

    # ==========================================================================
    # Overridden public methods:
    # ==========================================================================
    def run_aer_simulator(
        self, shots: int = 10000, show_plot: bool = True
    ) -> dict[str, int]:
        """
        Runs the Aer Simulator on the circuit that encodes
        the input vector and returns the simulation results.
        Also applies all the necessary measurements before
        running a simulation and shows a histogram of the
        simulation results if the 'show_plot' parameter
        is set to True. This method overrides the parent
        method to change the endianness of the qubit labels.

        Parameters:
        -----------
        shots: int
            The number of times a simulation is
            performed. Defaults to 10000.
        show_plot: bool
            Whether to show a histogram of the
            simulation results. Defaults to True.

        Returns:
        --------
        dict[str, int]
            The simulation counts with the correct qubit labels.
        """

        # Run the simulation and get its results by calling the
        # superclass's 'run_aer_simulator' method with 'False' for
        # the the parameter 'show_plot' to not display the plot
        # (because the plot would be incorrect due to reversed
        # endianness; the plot can be shown later in this method):
        simulation_counts = super().run_aer_simulator(shots, False).get_counts()

        # Initialize an empty dictionary to hold
        # the simulation results with reversed keys:
        reversed_simulation_counts = {}

        # Loop over the keys of the dictionary 'simulation_counts'
        # (they are bitstrings, representing the qubit states):
        for key in simulation_counts.keys():
            # Reverse the bitstring key to correct the
            # endianness by using the slicing method [::-1]:
            reversed_key = key[::-1]

            # Assign the count from the original simulation results to the
            # reversed key in the new dictionary 'reversed_simulation_counts':
            reversed_simulation_counts[reversed_key] = simulation_counts[key]

        # If the parameter 'show_plot' is 'True', plot the
        # histogram of the reversed simulation results:
        if show_plot is True:
            display(qiskit.visualization.plot_histogram(reversed_simulation_counts))

        # Return the desired dictionary:
        return reversed_simulation_counts

    # ==========================================================================
    # Public methods:
    # ==========================================================================
    def get_probabilities(self) -> np.ndarray:
        """
        Calculates and returns the probabilities of measuring each
        possible quantum state. This method first obtains the full
        statevector of the quantum circuit. Then, it creates a density
        matrix from the statevector. Next, it identifies the qubits that
        are not measured and traces out their states. Finally, it calculates
        the probabilities from the resulting partial trace matrix.

        Returns:
        --------
        probabilities: numpy.ndarray
            A list of probabilities corresponding to the possible
            outcomes of a measurement. The length of the list is 2^n,
            where n is the number of measured qubits. The sum of all
            probabilities in the list is equal to 1.
        """

        # Retrieve the statevector of the whole quantum circuit
        # (that contains both measured and unmeasured qubits):
        full_statevector = self.get_statevector()

        # Convert 'full_statevector' into a density matrix representation
        # (the class 'qiskit.quantum_info.states.statevector.Statevector'
        # contains the information to calculate the density matrix):
        density_matrix = DensityMatrix(full_statevector)

        # Create a list of indices for qubits that are not measured
        # in this encoding method (those all the qubits whose indices
        # do not equal a power of 2; the indexes of the measured
        # qubits are stored in the variable 'self._measured_qubits'):
        qubits_to_trace_out = [
            i for i in range(self._qubit_count) if i not in self._measured_qubits
        ]

        # Perform the partial trace operation to trace out the unmeasured
        # qubits (this operation creates a new density matrix of a smaller
        # dimension that only contains information about the measured qubits):
        partial_trace_matrix = partial_trace(density_matrix, qubits_to_trace_out)

        # Extract the probabilities of measuring each possible state
        # from the reduced density matrix (the probabilities can be
        # obtained from the diagonal of the density matrix):
        probabilities = partial_trace_matrix.probabilities()

        # Return the obtained probabilities with corrected endianness,
        # the method 'self.reverse_endianness()' is used for this purpose:
        return self._reverse_endianness(probabilities)
