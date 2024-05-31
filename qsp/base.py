# ==============================================================================
# Imports:
# ==============================================================================
from abc import ABC, abstractmethod
import qiskit
import qiskit.quantum_info
import qiskit_aer as Aer
import numpy as np
from IPython.display import display
from numbers import Real


# ==============================================================================
# Class:
# ==============================================================================
class QuantumStatePreparation(ABC):
    """
    This class acts as a base class for all implemented
    encoding methods. It contains code that is inherited
    by each class that implements an encoding method.

    Subclasses are required to implement all abstract
    methods, providing specific encoding algorithms
    for different encoding methods.

    Abstract methods:
    -----------------
    _validate_input_vector()
        Validates the input vector and initializes class attributes.
    _encode_input_vector()
        Encodes the input vector into the quantum circuit.
    """

    # ==========================================================================
    # Public properties:
    # ==========================================================================
    @property
    def input_vector(self) -> list[Real]:
        """
        Returns a copy of the vector that is
        encoded in the quantum circuit.

        Returns:
        --------
        list[numbers.Real]
            A copy of the vector that is
            encoded in the quantum circuit.
        """

        # Return a copy of the vector that is encoded in the quantum
        # circuit. A copy is returned so that the original encoded
        # vector can not be modified outside of this class.
        return self._input_vector.copy()

    # --------------------------------------------------------------------------
    @property
    def qubit_count(self) -> int:
        """
        Returns the number of qubits used in the quantum circuit.

        Returns:
        --------
        int
            The number of qubits used in the quantum circuit.
        """

        # Return the number of qubits used in the quantum circuit.
        # The variable '_qubit_count' is populated when
        # calling the '_validate_input_vector()' method.
        return self._qubit_count

    # --------------------------------------------------------------------------
    @property
    def quantum_circuit(self) -> qiskit.QuantumCircuit:
        """
        Returns the quantum circuit that encodes the input vector.

        Returns:
        --------
        qiskit.QuantumCircuit
            The quantum circuit that encodes the input vector.
        """

        # Return the quantum circuit used to encode the
        # input vector. A copy is not created, so the
        # circuit can be modified outside of this class.
        return self._quantum_circuit

    # --------------------------------------------------------------------------
    @property
    def measured_qubits(self) -> list[int]:
        """
        Returns a copy of the list of the measured qubits.
        Which qubits should be measured (to obtain the
        encoding results) depends on the chosen encoding
        method. All qubits should be measured when using
        the basis, angle, and amplitude encoding methods,
        while only some qubits should be measured when
        using the divide-and-conquer encoding method.

        Returns:
        --------
        list[int]
            A copy of the list of the qubits that
            should be measured to obtain the encoding results.
        """

        # Return a copy of the list of the measured qubits
        # so that the original list can not be modified outside
        # of this class. The variable '_measured_qubits' used
        # in the 'measure()' method is populated when calling
        # the '_validate_input_vector()' method.
        return self._measured_qubits.copy()

    # ==========================================================================
    # Constructor:
    # ==========================================================================
    def __init__(self, input_vector: list[Real]):
        """
        Initializes a new instance of the QuantumStatePreparation class with
        the chosen encoding method and encodes the input vector. An error is
        raised if the input vector is not valid for the chosen encoding method.

        Parameters:
        -----------
        input_vector: list[numbers.Real]
            The input vector to be encoded into a quantum circuit.

        Raises:
        -------
        ValueError
            If the input vector is empty or if it
            contains elements other than numbers.
        """

        # Raise an error if the input vector is empty:
        if len(input_vector) < 1:
            raise ValueError("The input vector must contain at least one element.")

        # Raise an error if the input vector
        # contains something other than real numbers:
        if not all(isinstance(element, Real) for element in input_vector):
            raise ValueError("The input vector must be a list of real numbers.")

        # Create a copy of the input vector so that it can be modified
        # within the class without affecting the original input vector
        # and save it to the '_input_vector' variable (the variable is
        # later used in the '_validate_input_vector()' and
        #'_encode_input_vector()' methods). The original 'input_vector'
        # parameter is later not modified/used anywhere.
        self._input_vector = input_vector.copy()

        # Calculate and save the norm of the input vector to
        # the '_input_vector_norm' variable (the variable
        # is later used in the '_validate_input_vector()'
        # and '_encode_input_vector()' methods):
        self._input_vector_norm = np.linalg.norm(self._input_vector)

        # Check whether the input vector is valid and initialize some
        # variables (such as '_qubit_count' and '_measured_qubits'
        # because these values may differ based on the encoding method
        # used) that will be used later (for example when the
        # '_encode_input_vector()' method is called):
        self._validate_input_vector()

        # Create a quantum register with the required number of
        # qubits (with the qubit count stored in the '_qubit_count'
        # variable calculated in the '_validate_input_vector()' method)
        # and set the name to 'q', so that i-th qubit is named 'q_i':
        quantum_register = qiskit.QuantumRegister(self._qubit_count, name="q")

        # Create a quantum circuit (with the variable 'quantum_register'
        # created above) and store in the '_quantum_circuit' variable
        # (the variable is used during the encoding process and
        # in various other methods) that can be retrieved by
        # the property 'quantum_circuit()' defined below:
        self._quantum_circuit = qiskit.QuantumCircuit(quantum_register)

        # Encode the input vector into the quantum
        # circuit based on the encoding method chosen:
        self._encode_input_vector()

    # ==========================================================================
    # Abstract methods:
    # ==========================================================================
    @abstractmethod
    def _validate_input_vector(self) -> None:
        """
        An abstract method that based on the chosen encoding method
        checks whether the input vector is valid and initializes
        some variables (retrievable by the properties above).

        Subclasses must override this method.
        """

        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def _encode_input_vector(self) -> None:
        """
        An abstract method that encodes the input vector
        into the quantum circuit. The encoding process
        depends on the type of the chosen encoding method.

        Subclasses must override this method.
        """

        pass

    # ==========================================================================
    # Public methods:
    # ==========================================================================
    def measure(self) -> None:
        """
        Removes any previous measurements from the
        circuit and adds new measurements to the qubits
        specified in the 'self._measured_qubits' list.

        Returns:
        --------
        None
        """

        # Remove any measurements previously added to the circuit:
        self._quantum_circuit.remove_final_measurements()

        # Create a classical register to hold the measurement results.
        # The quantum circuit's previous classical register was deleted
        # when the 'remove_final_measurements()' method was called.
        classical_register = qiskit.ClassicalRegister(len(self._measured_qubits))

        # Add the classical register to the circuit:
        self._quantum_circuit.add_register(classical_register)

        # Add a barrier to the circuit to visually separate
        # the measurements from the other gates:
        self._quantum_circuit.barrier()

        # Add the required measurements to the circuit. The
        # '_measured_qubits' list specifies which qubits
        # should be measured (to obtain the encoding results).
        # All qubits should be measured when using the basis,
        # angle, and amplitude encoding methods, while only
        # some qubits should be measured when using the
        # divide-and-conquer encoding method.
        self._quantum_circuit.measure(
            self._measured_qubits,
            [index for index in range(len(self._measured_qubits))],
        )

    # --------------------------------------------------------------------------
    def run_aer_simulator(
        self, shots: int = 10000, show_plot: bool = True
    ) -> qiskit.result.result.Result:
        """
        Runs the Aer Simulator on the circuit that encodes
        the input vector and returns the simulation results.
        Also applies all the necessary measurements before
        running a simulation and shows a histogram of the
        simulation results if the 'show_plot' parameter
        is set to True.

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
        qiskit.result.result.Result
            The simulation results.
        """

        # Apply all the necessary measurements
        # before running a simulation:
        self.measure()

        # Create a basic 'AerSimulator' instance:
        simulator = Aer.AerSimulator()
        # Perform the simulations and save the results:
        simulation_results = simulator.run(self._quantum_circuit, shots=shots).result()

        # Show a histogram of the simulation
        # results if 'show_plot' is set to True:
        if show_plot is True:
            # Get the counts from the simulation results:
            counts = simulation_results.get_counts()
            # Show a histogram of the counts:
            display(qiskit.visualization.plot_histogram(counts))

        return simulation_results

    # --------------------------------------------------------------------------
    def get_statevector(self) -> qiskit.quantum_info.states.statevector.Statevector:
        """
        Returns the statevector of the quantum
        circuit that encodes the input vector.

        Returns:
        --------
        qiskit.quantum_info.Statevector
            The statevector of the quantum circuit.
        """

        # Remove any measurements in case they were previously
        # added to the circuit (the 'Statevector.from_instruction'
        # method does not work with measurements present in a circuit):
        self._quantum_circuit.remove_final_measurements()

        # Get and return the statevector of the quantum circuit:
        return qiskit.quantum_info.Statevector.from_instruction(self._quantum_circuit)

    # ==========================================================================
    # Private methods:
    # ==========================================================================
    def _normalize_input_vector(self) -> None:
        """
        Normalizes the input vector if it is not already normalized.

        Raises:
        -------
        ZeroDivisionError
            If the norm of the input vector is 0 (the input vector
            can not be normalized by dividing it by zero).

        Returns:
        --------
        None
        """

        # Check whether the input vector is already normalized (due to the
        # nature of floating point numbers, the norm might not equal 1 even
        # though the vector is normalized, so use 'np.isclose' method instead):
        if not np.isclose(self._input_vector_norm, 1):
            # The input vector is not normalized.

            # Raise an error if the norm of the input vector is 0,
            # because the input vector can not be normalized:
            if self._input_vector_norm == 0:
                raise ZeroDivisionError("The norm of the input vector is 0.")

            print(
                "The input vector with the norm of",
                self._input_vector_norm,
                "has been normalized.",
            )

            # Normalize the input vector:
            self._input_vector /= self._input_vector_norm

            # Update the norm of the input vector (should equal 1):
            self._input_vector_norm = np.linalg.norm(self._input_vector)
