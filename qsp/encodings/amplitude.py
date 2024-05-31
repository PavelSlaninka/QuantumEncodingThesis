# ==============================================================================
# Imports:
# ==============================================================================
from qsp.base import QuantumStatePreparation
import numpy as np
import math
import qiskit.quantum_info
from numbers import Real


# ==============================================================================
# Class:
# ==============================================================================
class AmplitudeEncoding(QuantumStatePreparation):
    """
    The class 'AmplitudeEncoding' implements the amplitude
    encoding method for encoding a normalized input vector
    into a quantum state. The amplitude encoding method
    encodes the elements of the input vector into the quantum
    state amplitudes qubits using multi-controlled gates 'ry',
    where the angles used in the rotations are calculated from
    the input vector elements, and the number of rotations
    and qubits depend on the length of the input vector.

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
    >>> from qsp import AmplitudeEncoding
    >>> angle_encoding = qsp.AngleEncoding([-0.5, 0.5, -0.5, 0.5])
    >>> angle_encoding.get_statevector()
    Statevector([-0.5+0.j,  0.5+0.j, -0.5+0.j,  0.5+0.j], dims=(2, 2))

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

        # Get the length of the input vector:
        length = len(self._input_vector)

        # Calculate the number of qubits required to encode the
        # input vector (in the amplitude encoding method, it is
        # logarithmic (log2) compared to the length of the input
        # vector; the 'ceil' function is used for when the input
        # vector length does not equal a power of two):
        self._qubit_count = math.ceil(math.log2(length))

        # In case the input vector only contains one
        # element, the '_qubit_count' was incorrectly
        # calculated as being 0, so set it to 1 manually:
        if self._qubit_count == 0:
            self._qubit_count = 1

        # Create a list of the qubits that should be measured
        # to obtain the encoding results (in the amplitude
        # encoding method, every qubit should be measured):
        self._measured_qubits = [i for i in range(self._qubit_count)]

        # Normalize the input vector if its norm is not equal
        # to one (it is important in the amplitude encoding
        # method for the input vector to be normalized):
        super()._normalize_input_vector()

    # --------------------------------------------------------------------------
    def _encode_input_vector(self) -> None:
        """
        Encodes the input vector using the amplitude encoding method.
        Multi-controlled gates 'ry' are utilized to encode the input vector.
        The encoding process is explained step-by-step in the comments.

        Returns:
        --------
        None
        """

        # Calculate the angles that need to be used in multi-controlled rotations:
        angles = AmplitudeEncoding.calculate_amplitude_angles(
            self.input_vector, self._qubit_count
        )

        # Apply the gate 'ry' to the last qubit using the first angle:
        self._quantum_circuit.ry(angles[0], self._qubit_count - 1)

        # Create a variable to keep track of the index
        # of the angle that needs to be used next:
        angle_index = 1

        # Iterate through all qubits of the quantum circuit except for
        # the last qubit (the last qubit already has the gate 'ry' applied
        # to it and it should not have any other gates targetting it):
        for index in range(self._qubit_count - 1):
            # Each iteration of this outer for loop represents a binary
            # tree level. One gate 'ry' has been used in the first binary
            # tree level. This loop starts from the second binary tree
            # level (the levels are indexed from zero in this case):
            level = index + 1

            # In each binary tree depth level,
            # 'level'^2 rotational multi-controlled gates are used:
            y_gates_on_this_level = int(math.pow(2, level))

            # Each gate in a given binary tree level targets the same qubit, where
            # the target qubit is calculated in the following way (so this starts
            # from the second to the last qubit and goes to the first qubit):
            target_qubit = self._qubit_count - index - 2

            # Each gate in a given level uses the same control qubits.
            # The indexes of the control qubits are calculated (they range
            # from the qubit that follows the target qubit to the last qubit):
            control_qubits = [i for i in range(target_qubit + 1, self._qubit_count)]

            # Iterate through the number of gates that are needed
            # to be applied at this level (in each iteration of this
            # inner for loop, one rotational gate is applied):
            for number in range(y_gates_on_this_level):
                # Retrieve the angle that needs to be
                # used in the current rotation:
                angle_to_encode = angles[angle_index]

                # Only proceed to apply the gate if the angle is not equal to
                # zero (when zero is used as the angle in a rotation, such a
                # rotation does not alter the state of a qubit in any way):
                if angle_to_encode != 0:
                    # The last thing before the gate can be applied is that the
                    # control states of the control qubits need to be determined.
                    # This can be done by converting the gate index 'number' into
                    # its binary representation (the string needs to be padded
                    # with leading zeros so that its length equals the
                    # number of control qubits):
                    control_states = bin(number)[2:].zfill(level)

                    # Reverse the string so that it corresponds to the endianness
                    # used in this encoding method implementation (where the first
                    # character represents the first control qubit's control state):
                    control_states = list(reversed(control_states))

                    # Apply the correct multi-controlled gate 'ry':
                    self._mcry_with_control_states(
                        angle_to_encode, target_qubit, control_qubits, control_states
                    )

                # An angle was used in the rotation, so increment
                # the angle index so that in the following rotation,
                # the next angle will be used:
                angle_index += 1

    # ==========================================================================
    # Private methods:
    # ==========================================================================
    def _mcry_with_control_states(
        self, angle, target_qubit, control_qubits, control_states
    ):
        """
        Applies a multi-controlled gate 'ry' to the target qubit,
        conditioned on the control states of the control qubits.

        Parameters:
        -----------
        angle: float
            The angle that is used in the gate
            'mcry' applied to the 'target_qubit'.
        target_qubit: int
            The index of the qubit that needs to be
            conditionally rotated using the gate 'mcry'.
        control_qubits: list[int]
            A list of the indices of the qubits that
            serve as control qubits in the gate 'mcry'.
        control_states: list[str]
            A list of the control states, where each element
            is a character, either '0' or '1', indicating the
            control state of the corresponding control qubit.
            The 'i'-th element of the list serves as a control state
            of the 'i'-th qubit stored in the 'control_qubits' list.

        Returns:
        --------
        None
        """

        # The first step is to flip all the control qubits with a
        # control state that is equal to the state |0>. The state
        # |1> is the default state in control gates. Nothing needs
        # to be done if a control qubit has |1> as its control state.
        # However, if a qubit's control state is |0>, then the qubit needs
        # to be flipped using the gate 'x' before it can be used in the
        # controlled gate as a control qubit conditioned on the state |0>.

        # Create an empty list to store the qubits
        # that need to have the gate 'x' applied to them:
        qubits_x_gate = []

        # Iterate over all control qubits and control states.
        # The length of both lists is the same. I-th 'qubit'
        # has i-th 'state' as its control state:
        for qubit, state in zip(control_qubits, control_states):
            # If the control state is |0>, then add
            # the qubit index to the 'qubits_x_gate' list:
            if state == "0":
                qubits_x_gate.append(qubit)

        # Apply the gate 'x' to the control qubits
        # that need to have this gate applied to them,
        # do nothing if no gates 'x' need to be applied:
        self._quantum_circuit.x(qubits_x_gate) if len(qubits_x_gate) > 0 else None

        # Apply the multi-controlled gate 'ry' to the target qubit,
        # conditioned on the control states of the control qubits
        self._quantum_circuit.mcry(angle, control_qubits, target_qubit)

        # If a control qubit had the gate 'x' applied to it,
        # then another gate 'x' needs to be added to it in
        # order to revert the qubit into its initial state,
        # leaving it unmodified. It is important not to modify
        # the control qubits to retain the desired effect of
        # the controlled gates 'ry' that were potentially
        # applied to them in previous function calls.

        # Apply the gate 'x' to all qubits
        # stored in the 'qubits_x_gate' list
        # do nothing if no gates 'x' need to be applied:
        self._quantum_circuit.x(qubits_x_gate) if len(qubits_x_gate) > 0 else None

    # ==========================================================================
    # Overridden public methods:
    # ==========================================================================
    def get_statevector(self) -> qiskit.quantum_info.states.statevector.Statevector:
        """
        Returns the statevector of the quantum
        circuit that encodes the input vector.

        This method overrides the parent method
        'super().get_statevector()' to account for
        cases when the input vector's length is
        not equal to a power of 2.

        Returns:
        --------
        qiskit.quantum_info.states.statevector.Statevector
            The statevector of the quantum circuit.
        """

        # Get the statevector of the whole quantum
        # circuit by calling the parent method:
        statevector = super().get_statevector()

        # Check if the input vector's length is not equal to a power of 2:
        if len(self._input_vector) != len(statevector.data):
            # The input vector's length is not equal to a power of 2. An input
            # vector's length must equal a power of two to fully utilize a quantum
            # circuit when using the amplitude encoding method. During the encoding
            # process, the input vector was padded with zeros. The last
            # 'len(statevector.data) - len(self._input_vector)' angles used in the
            # controlled gates 'ry' do not affect the statevector, so the last
            # 'len(statevector.data) - len(self._input_vector)' elements of the
            # statevector are zeros (because the input vector was padded with zeros).
            # The amplitude encoding method requires the returned statevector's length
            # to be the same as the input vector's length, so the last
            # 'len(statevector.data) - len(self._input_vector)' zeros
            # must be removed from the statevector.

            # Extract the statevector data while only keeping the required elements:
            statevector_required_data = statevector.data[0 : len(self._input_vector)]

            # Create a new statevector instance with only the required elements:
            statevector = qiskit.quantum_info.Statevector(statevector_required_data)

        return statevector

    # ==========================================================================
    # Private static methods:
    # ==========================================================================
    @staticmethod
    def _pad_input_vector(input_vector: np.ndarray, qubit_count: int) -> np.ndarray:
        """
        Returns the original input vector. In case the input
        vector's length is not equal to 2^self._qubit_count
        (a power of two 2^n, where n > 0), a new vector is
        created and returned by appending enough zeros to the
        end of the input vector to make its length equal to the
        nearest power of 2 larger than the initial length of the
        input vector, leaving the original input vector unmodified.

        Parameters:
        -----------
        input_vector: numpy.ndarray
            The input vector to be padded.
        qubit_count: int
            The number of qubits in the quantum circuit required
            by the encoding method to encode the input vector.

        Returns:
        --------
        npumpy.ndarray
            The input vector or a copy of the input
            vector padded with zeros at the end to
            make its length equal to 2^self._qubit_count.

        Note:
        -----
        This method is static because it is used in the static
        method 'calculate_amplitude_angles' defined below.
        """

        # Get the length of the input vector:
        length = len(input_vector)

        # Create a new variable to store the vector
        # that will be returned by this method:
        return_vector = input_vector

        # Check if the 'length' equals 2^self._qubit_count:
        if length != 2**qubit_count:
            # The input vector's length is not equal to 2^n,
            # where n > 0. A new vector is created (so that the
            # original input vector is not modified) by appending
            # enough zeros at the end of the input vector to increase
            # the size of the returned vector to the nearest power of
            # two larger than the current input vector's length:
            return_vector = np.append(
                input_vector,
                # '(2**self._qubit_count) - length' is the number
                # of zeros required to increase the length of the
                # returned vector to the nearest power of 2:
                [0] * ((2**qubit_count) - length),
            )

        return return_vector

    # ==========================================================================
    # Public static methods:
    # ==========================================================================
    @staticmethod
    def calculate_amplitude_angles(
        input_vector: list[Real], qubit_count: int
    ) -> np.ndarray:
        """
        Uses an input vector to calculate the desired angles used in
        the rotations 'ry'. The input vector is first padded to a
        length that is a power of 2 if necessary. The angles are then
        calculated based on the probabilities derived from the squared
        elements of the padded input vector. The angle calculation
        process is explained step-by-step in the comments.

        Parameters:
        -----------
        input_vector: list[numbers.Real]
            A vector containing the input vector elements that
            need to be used to calculate the desired angles.
        qubit_count: int
            The number of qubits in a quantum circuit required
            by the encoding method to encode the input vector.

        Returns:
        --------
        npumpy.ndarray
            A vector containing the desired
            angles used in the rotations 'ry'.

        Note:
        -----
        This method is static and public because this angle calculation
        algorithm is also used in the divide-and-conquer encoding method 
        implemented in the file 'divide_and_conquer_encoding.py'.
        """

        # This angle calculation method only works for vectors of
        # length equal to a power of 2 with a minimal length of 2,
        # so pad the input vector if needed:
        vector = AmplitudeEncoding._pad_input_vector(input_vector, qubit_count)

        # Get the length of the (padded) input vector, it is a power of two:
        vector_length = len(vector)

        # Get the vector of probabilities of measuring each possible
        # state by squaring each element of the (padded) vector:
        probabilities = np.power(vector, 2)

        # Create an empty list of lists, where the number of lists
        # quals qubit count minus one. This represents a binary tree
        # structure. Each inner list element is a sum of multiple
        # elements of the 'probabilities' list in the following way:
        # - the first binary tree level (the first list) contains two
        #   elements, each element sums half of the 'probabilities'
        #   list, the first element sums the first half of the
        #   'probabilities' list, the second element sums the
        #   second half of the 'probabilities' list
        # - the first binary tree level (the second list) contains
        #   four elements, where the first element sums the first
        #   quarter of the 'probabilities' lis, etc...
        # - the third binary tree level (the third list) contains
        #   eight elements, where each element sums the corresponding
        #   eighth of the 'probabilities' list
        # - etc...
        # - the last binary tree level (the last list)
        #   contains vector_length/2 elements, each element sums the
        #   corresponding pair of the 'probabilities' list elements
        angles_probabilities = [[] for _ in range(qubit_count - 1)]

        # Create an empty list to store the calculated angles;
        # this list is returned at the end of this method:
        angles = []

        # Iterate over all qubit indexes, each represents a binary tree
        # level (in each level, 'binary_level'^2 angles are calculated,
        # this means that in the amplitude encoding method,
        # 'binary_level'^2 gates 'mcry' are applied to the
        # corresponding qubit characterized by the 'binary_level'):
        for binary_level in range(qubit_count):
            # In each binary tree level, 'binary_level'^2 angles are
            # calculated. Each angle requires
            # (vector_length//(2^(binary_level+1))
            # elements of the 'probabilities' list to be calculated:
            probabilities_in_angle = vector_length // (2 ** (binary_level + 1))

            # Create two variables 'counter' and 'probability_index_start_in_angle'.
            # The variable 'counter' goes form from 0 to
            # ('vector_length' / 'probabilities_in_angle') and the variable
            # 'probability_index_start_in_angle' goes from 0 to 'vector_length'.
            # In each iteration:
            # - 'counter' is incremented by one,
            # - 'probability_index_start_in_angle' is
            #   incremented by 'probabilities_in_angle'.
            # The variable 'probability_index_start_in_angle' denotes the index
            # of the first element of the 'probabilities' list that needs to be summed.
            for counter, probability_index_start_in_angle in enumerate(
                range(0, vector_length, probabilities_in_angle)
            ):
                # Create a sum, where 'probabilities_in_angle' elements of
                # the list 'probabilities' are summed, starting from the
                # 'probability_index_start_in_angle'-th element.
                angle_probabilities_summed = np.sum(
                    probabilities[
                        probability_index_start_in_angle : probability_index_start_in_angle
                        + probabilities_in_angle
                    ]
                )

                # Append the summed probabilities to the tree structure on the correct
                # binary level (check for 'binary_level' being in the bounds of the list
                # 'angles_probabilities'; it is not needed to fill out and remember
                # the last binary tree level, so that is why only the first
                # ('qubit_count' - 1) levels are filled):
                angles_probabilities[binary_level].append(
                    angle_probabilities_summed
                ) if binary_level < qubit_count - 1 else None

                # The summed probabilities are now appended to the tree, so, now,
                # the angle calculation itself can start. 'binary_level'^2 angles
                # are calculated in each binary tree level, but in each level of
                # the 'angles_probabilities', there are ('binary_level'^2 * 2)
                # probabilities stored. So only every second of these probabilities
                # is used to calculate the angles in a level. This is because if the
                # probabilities in a level are divided into pairs, the first element
                # of a pair can be used in the cosine function and the second element
                # of a pair can be used in the sine function. In this implementation,
                # the cosine is used, so each even-indexed probability is used, thus
                # each odd-indexed probability can be skipped:
                if counter % 2 == 1:
                    continue

                # Before the angle can be calculated, 'angle_probabilities_summed'
                # needs to be normalized in accordance with its parent stored in the
                # binary tree. This is because each angle is calculated based on a
                # probability of a qubit collapsing into the state |0> (because the
                # cosine function is used) conditioned on the state of the preceding
                # qubits. For example:
                # - The first angle is calculated based on the probability of the first
                #   qubit collapsing into the state |0> when measured, meaning the
                #   probability of measuring the state |*0> is used in the first angle
                #   calculation. For this state, the first half of the 'probabilities'
                #   list is used. So in this case, no normalization is needed.
                #   The summed first half and the summed second half are
                #   appended to the first binary tree level.
                # - In the second binary tree level, two angles are calculated, with
                #   the two used states being |*00> and |*01>. To calculate the
                #   probabilities of measuring these two states, the two summed values
                #   are used that are stored in the first binary tree level (they are
                #   used in the division operation, where they divide the summed
                #   probabilities created in the second binary tree level; more
                #   specifically, the summed first quarter of the 'probabilities'
                #   is divided by the summed first half, and the third quarter
                #   is divided by the second half).
                if binary_level > 0:
                    # Get the correct parent summed probabilities from the binary tree:
                    parent_angle_probabilities = angles_probabilities[binary_level - 1][
                        counter // 2
                    ]

                    # Only perform the division if it is defined:
                    if parent_angle_probabilities != 0:
                        angle_probabilities_summed /= parent_angle_probabilities

                # Now the angle itself can be calculated using the cosine
                # function. This is used because the rotations 'ry' are
                # performed, which is cos(theta/2). When cos(theta/2) is
                # squared, it must equal 'angle_probabilities_summed'.
                angle_to_append = 2 * np.arccos(np.sqrt(angle_probabilities_summed))

                # In the angle calculation formula used above, probabilities are used.
                # Probabilities are always positive numbers. This means that regardless
                # of the sign of an input vector element, the sign is always lost because
                # the element is squared during the summation. This means that the
                # calculated angle always belongs to the interval [0, pi]. And this in
                # turn means that after the rotations 'ry' are performed, the resulting
                # state vector elements are always positive numbers. However, the input
                # vector elements can be negative numbers. So, if an input vector element
                # is negative, it needs to be reflected in the resulting state vector,
                # where its corresponding element must also be negative. To achieve such
                # a result, the following procedure must be performed, where the calculated
                # angle is adjusted according to the input vector elements' signs:
                if binary_level == qubit_count - 1:
                    # It is sufficient to only perform this adjustment for the angles
                    # calculated in the last binary tree level. In the last level, the
                    # input vector elements can be divided into pairs, where
                    # 'probability_index_start_in_angle' is the index of the first element
                    # in a pair, and ('probability_index_start_in_angle' + 1) is the index
                    # of the second element in a pair.

                    # If the second element of a pair is negative, the angle is negated:
                    if vector[probability_index_start_in_angle + 1] < 0:
                        angle_to_append *= -1

                    # If the first element of a pair is negative,
                    # the angle is subtracted from 2*pi:
                    if vector[probability_index_start_in_angle] < 0:
                        angle_to_append = 2 * np.pi - angle_to_append

                # Append the calculated angle to the list 'angles':
                angles.append(angle_to_append)

        return angles
