"""
This file contains functions that are used in multiple Notebooks
located in this directory. The reason the code in this file
is not present in those Notebooks is to prevent excessive
length of the Notebooks and to avoid duplicate code.
"""

# ==============================================================================
# Imports:
# ==============================================================================
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from qiskit.result.counts import Counts
from IPython.display import display
from numbers import Number
import warnings

with warnings.catch_warnings():
    # Catch and ignore warnings from the following import
    # call so that they don't appear in the Notebook output
    # (a warining appears on Windows machines saying the
    # 'kahypar' package can not be imported):
    warnings.filterwarnings("ignore")

    from quimb import rand_haar_state


# ==============================================================================
# Functions:
# ==============================================================================
def generate_haar_random_vector(vector_length: int) -> np.ndarray:
    """
    Generates a normalized random float vector of the specified
    length using the Haar distribution and returns it in the form
    of a numpy array. The elements of this vector are designed to
    be uniformly distributed on the Bloch sphere after being encoded.
    The 'quimb' library is used to generate the state.

    Parameters
    ----------
    vector_length: int
        The length of the generated random vector.

    Returns
    -------
    numpy.ndarray
        A normalized random vector of the length 'vector_length'
        generated according to the Haar measure. The type of
        the list elements is 'numpy.float64'.
    """

    # Generate a random state of dimension 'vector_length'
    # according to the Haar measure ('float' as a data type
    # is chosen because it is sufficient to demonstrate the
    # encoding methods, but 'complex' can also be used):
    generated_rand_haar_state = rand_haar_state(vector_length, dtype=float)

    # Return a flattened numpy array:
    return np.asarray(generated_rand_haar_state).flatten()


# ------------------------------------------------------------------------------
def add_states_if_missing(counts: Counts) -> None:
    """
    Adds all possible states as keys to the 'counts' dictionary
    (with the values being zero) if they are not present.

    Parameters:
    -----------
    counts: qiskit.result.counts.Counts
        The counts dictionary to add the missing states to.

    Returns:
    --------
    None
    """

    # Get the number of measured qubits by taking the length
    # of the first key of the 'counts' dictionary (the
    # keys are binary strings representing the states):
    qubit_count = len(list(counts.keys())[0])

    # Iterate through all possible states:
    for index in range(2**qubit_count):
        # Create a key representing the state by
        # converting the integer 'index' to a binary string:
        key = format(index, "0{}b".format(qubit_count))

        # If the key 'key' is not present in the 'counts' dictionary,
        # add it to the 'counts' dictionary and set the count value to 0:
        if key not in counts:
            counts[key] = 0


# ------------------------------------------------------------------------------
def results_for_qubits(counts: Counts) -> list[int]:
    """
    Uses the 'counts' dictionary from a simulation and,
    for each individual qubit, calculates how many times
    it collapsed to the state |1>. It then returns a list
    of the results.

    Parameters:
    -----------
    counts: qiskit.result.counts.Counts
        The counts dictionary containing the measurement results.

    Returns:
    --------
    list[int]
        A list containing the number of collapses to the
        state |1> for each qubit. The order of the list
        corresponds to the order of the qubits in the circuit.
    """

    # Get the number of measured qubits by taking the length
    # of the first key of the 'counts' dictionary (the
    # keys are binary strings representing the states):
    qubit_count = len(list(counts.keys())[0])

    # Initialize a list of zeros to store the results for each qubit:
    qubits_results = [0] * qubit_count

    # Iterate through all states in the 'counts' dictionary:
    for key in counts:
        # Iterate through each qubit of the state. The binary string
        # representing the state is reversed so that i-th character
        # of the string corresponds to the i-th qubit in the circuit:
        for index, char in enumerate(reversed(key)):
            # If the state of the qubit is |1>,
            # add the count value to the result:
            if char == "1":
                qubits_results[index] += counts[key]

    return list(qubits_results)


# ------------------------------------------------------------------------------
def show_qubit_results_table(
    shots: int, expected_probabilities: list[float], qubit_results: list[int]
) -> None:
    """
    Takes in the number of shots of the simulation, the
    expected probabilities of each qubit collapsing to the
    state |1>, and the actual measurement results of each
    qubit collapsing to the state |1> (obtained by calling
    the 'results_for_qubits' function), and shows a table
    containing the expected and actual number of the
    collapses to the state |1> for each qubit.

    Parameters:
    -----------
    shots: int
        The number of shots used in the simulation.
    expected_probabilities: list[float]
        A list of the expected probabilities of
        measuring each qubit in the state |1>.
    qubit_results: list[int]
        A list of the actual measurement results of the
        collapses to the state |1> for each qubit,
        obtained from the 'results_for_qubits' function.

    Returns:
    --------
    None
    """

    # Multiply the expected probabilities by the number of
    # shots to get the expected number of collapses to the state |1>:
    expected_probabilities_modified = np.round(
        np.multiply(expected_probabilities, shots)
    ).astype(int)

    # Create a dictionary to store the data for the results table:
    data = {
        "Expected count for |1>": expected_probabilities_modified,
        "Actual count for |1>": qubit_results,
    }

    # Add the 'data' to DataFrame:
    data_frame = DataFrame(data)

    # Rename the index axis with a label:
    data_frame = data_frame.rename_axis("Qubit").reset_index()

    # Apply a style to the 'data_frame' to format it as a colorful table
    # and show it (colormap is chosen to be 'Blues' and 'subset' must be
    # set to include only the columns with the counts (such columns are
    # stored as keys of the 'data' dictionary); otherwise, colors would
    # not be applied to the table because the 'Qubit' column was added
    # with the 'rename_axis("Qubit").reset_index()' command):
    display(
        data_frame.style.background_gradient(cmap="Blues", subset=list(data.keys()))
    )


# ------------------------------------------------------------------------------
def calculate_state_probabilities(
    expected_probabilities: list[float],
) -> list[float]:
    """
    Calculates the probabilities of measuring each possible composite
    state given the 'expected_probabilities' list containing the
    expected probabilities of each qubit collapsing to the state |1>.

    Parameters:
    -----------
    expected_probabilities: list[float]
        A list of the expected probabilities of
        each qubit collapsing to the state |1>.

    Returns:
    --------
    list[float]
        A list of probabilities of measuring each possible composite state.
        The list is sorted such that its i-th element represents the
        probability of measuring the quantum state in the state described
        by the i-th binary string, where binary strings of length 'n'
        (where 'n' equals the squared length of the 'expected_probabilities'
        list) are ordered by their least significant bit.
    """

    # Get the number of qubits (in this function
    # it is assumed that the number of qubits is equal
    # to the length of the 'expected_probabilities' list):
    qubit_count = len(expected_probabilities)

    # Initialize a list to store the
    # probabilities of measuring each possible state:
    state_probabilities = []

    # Iterate through each possible state and
    # calculate the probability of measuring it:
    for binary_state in range(2**qubit_count):
        probability = np.prod(
            # The probability of a composite event is the product
            # of the probabilities of its individual components:
            [
                # The '(binary_state >> qubit) & 1' condition is True if the
                # i-th qubit (where the index 'i' is defined by the 'qubit'
                # variable) of the state represented by the binary string
                # 'binary_state' is equal to 1. Otherwise, the condition evaluates
                # to False. The 'probability' variable contains the probability
                # of measuring the qubit 'qubit' in the state |1>, so
                # '1 - probability' is the probability of measuring the qubit
                # 'qubit' in the state |0>. If the condition evaluates to False
                # (meaning that the qubit 'qubit' in the composite state is in the
                # state |0>), '1 - probability' must be used instead of 'probability':
                (probability if (binary_state >> qubit) & 1 else (1 - probability))
                # Iterate through probabilities of
                # measuring each qubit in the state |1>:
                for qubit, probability in enumerate(expected_probabilities)
            ]
        )

        # Add the calculated probability of measuring the
        # composite state to the list of probabilities:
        state_probabilities.append(probability)

    return state_probabilities


# ------------------------------------------------------------------------------
def calculate_qubit_probabilities(
    expected_probabilities: list[float],
) -> list[float]:
    """
    Calculates the probabilities of measuring each qubit in the
    state |1> state given the 'expected_probabilities' list containing
    the expected probabilities of measuring each composite state.

    Parameters:
    -----------
    expected_probabilities: list[float]
        A list of the expected probabilities
        of measuring each possible state.

    Returns:
    --------
    list[float]
        A list of probabilities of measuring each qubit in the
        state |1>. The list is sorted such that its i-th element
        represents the probability of measuring the i-th qubit
        of the quantum circuit in the state |1>.
    """

    # Get the number of qubits (in this function it is assumed
    # that the number of qubits is equal to 'n', where '2^n'
    # is the length of the 'expected_probabilities' list):
    qubit_count = int(np.log2(len(expected_probabilities)))

    # Initialize a list of zeros to store the results for each qubit:
    qubits_results = [0.0] * qubit_count

    # Iterate through all states:
    for state_index in range(len(expected_probabilities)):
        # Create a key representing the state by
        # converting the integer 'index' to a binary string:
        key = format(state_index, "0{}b".format(qubit_count))

        # Iterate through each qubit of the state. The binary string
        # representing the state is reversed so that i-th character
        # of the string corresponds to the i-th qubit in the circuit:
        for qubit_index, char in enumerate(reversed(key)):
            # If the state of the qubit is |1>,
            # add the count value to the result:
            if char == "1":
                qubits_results[qubit_index] += expected_probabilities[state_index]

    return list(qubits_results)


# ------------------------------------------------------------------------------
def compare_probabilities_expected_actual(
    state_probabilities_expected: list[float], state_probabilities_actual: list[float]
) -> None:
    """
    Compares two lists of probabilities of measuring each
    possible state and raises an error if they are not equal.

    Parameters:
    -----------
    state_probabilities_expected: list[float]
        A list of the expected probabilities of measuring each possible state.
    state_probabilities_actual: list[float]
        A list of the actual probabilities of measuring each possible state.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If the two lists of probabilities are not equal.
    """

    # Initialize a bool variable indicating
    # whether the probabilities are the same to True:
    is_same = True

    # Iterate through all the possible states:
    for index in range(len(state_probabilities_expected)):
        if not np.isclose(
            state_probabilities_expected[index],
            state_probabilities_actual[index],
            rtol=1.0e-9,
        ):
            # If the expected and actual probabilities are
            # unequal, set the variable to False and break the loop:
            is_same = False
            break

    # Print the result if the probabilities are the same; otherwise, raise an error:
    if is_same:
        print("The expected probabilities are the same as the actual probabilities.")
    else:
        raise ValueError(
            "The expected probabilities are not the same as the actual probabilities."
        )


# ------------------------------------------------------------------------------
def compare_results_simulation_plot(
    shots: int,
    counts: Counts,
    state_probabilities_expected: list[float],
    state_probabilities_actual: list[float],
) -> None:
    """
    Creates and shows a bar plot comparing the following data:
      1. The simulation measurement results contained in the 'counts' parameter.
      2. The actual probabilities of measuring each possible state obtained
        from the statevector of the quantum system after applying all the gates.
      3. The expected probabilities of measuring each possible state
        obtained from the generated input vector


    Parameters:
    -----------
    shots: int
        The number of shots used in the simulation.
    counts: qiskit.result.counts.Counts
        The counts dictionary containing the simulation measurement results.
    state_probabilities_expected: list[float]
        A list of the expected probabilities of measuring each possible state.
    state_probabilities_actual: list[float]
        A list of the actual probabilities of measuring each possible state.

    Returns:
    --------
    None
    """

    # Extract the keys (state labels) from the 'counts' dictionary:
    keys_counts = list(counts.keys())

    # Sort the keys based on the least significant bit:
    sorted_keys = sorted(keys_counts, key=lambda x: int(x, 2))

    # Rearrange values from the 'counts' dictionary based on the sorted keys:
    sorted_values = [counts[key] for key in sorted_keys]

    # Create a figure and a primary y-axis:
    _, axis_1 = plt.subplots(figsize=(8, 6))

    # Calculate the horizontal positions of the bars and set the xticks positions:
    bar_positions_main = np.arange(len(sorted_keys))
    plt.xticks(bar_positions_main, sorted_keys)

    # Define the width of each bar:
    bar_width = 0.2

    # Plot the data from the 'counts' simulation
    # results as bars on the primary y-axis:
    axis_1.bar(
        bar_positions_main - bar_width,
        sorted_values,
        width=bar_width,
        color="r",
        label="Simulation counts\n(simulator)",
    )

    # Create a secondary y-axis that shares the same x-axis:
    axis_2 = axis_1.twinx()

    # Plot the actual probabilities data as bars on the secondary y-axis:
    axis_2.bar(
        bar_positions_main,
        state_probabilities_actual,
        width=bar_width,
        color="darkgreen",
        label="Actual probabilities\n(statevector)",
        align="center",
    )

    # Plot the expected probabilities data as bars on the secondary y-axis:
    axis_2.bar(
        bar_positions_main + bar_width,
        state_probabilities_expected,
        width=bar_width,
        color="lightgreen",
        label="Expected probabilities\n(input vector)",
        align="center",
    )

    # Find the maximum values of the simulation counts, actual
    # probabilities and expected probabilities (the simulation
    # counts need to be normalized to compare them to the probabilities):
    max_counts = np.max(np.divide(np.asarray(sorted_values, dtype=np.float64), shots))
    max_actual = np.max(state_probabilities_actual)
    max_expected = np.max(state_probabilities_expected)

    # Find the maximum value out of the three maximum
    # values above and add 0.1 as a padding for the y-axes.
    # This will be used to align the height of the
    # secondary y-axis with the height of the primary y-axis.
    max = np.max([max_counts, max_actual, max_expected]) + 0.1

    # Set the range of the y-axes as the maximum value defined above.
    # The range of the primary y-axis needs to be
    # scaled by the number of shots of the simulation.
    axis_1.set_ylim(0, max * shots)
    axis_2.set_ylim(0, max)

    # Add a title and labels, and show the plot:
    axis_1.set_xlabel("State")
    axis_1.set_ylabel("Count", color="r")
    axis_2.set_ylabel("Probability", color="g")
    axis_1.legend(loc="upper left")
    axis_2.legend(loc="upper right")
    plt.title("Expectations and results")
    plt.show()


# ------------------------------------------------------------------------------
def compare_results_experiment_plot(
    shots: int,
    counts_simulation: Counts,
    counts_experiment: Counts,
    state_probabilities_expected: list[float],
) -> None:
    """
    Creates and shows a bar plot comparing the following data:
      1. The simulation measurement results
        contained in the 'counts_simulation' parameter.
      2. The experiment measurement results contained in the 'counts_experiment'
        parameter obtained by running the circuit on a real quantum computer.
      3. The expected probabilities of measuring each possible state
        obtained from the generated input vector


    Parameters:
    -----------
    shots: int
        The number of shots used in the simulation.
    counts_simulation: qiskit.result.counts.Counts
        The counts dictionary containing the simulation measurement results.
    counts_experiment: qiskit.result.counts.Counts
        The counts dictionary containing the experiment measurement results.
    state_probabilities_actual: list[float]
        A list of the actual probabilities of measuring each possible state.

    Returns:
    --------
    None
    """

    # Extract the keys (state labels) from the 'counts_simulation' dictionary:
    keys_counts = list(counts_simulation.keys())

    # Sort the keys based on the least significant bit:
    sorted_keys = sorted(keys_counts, key=lambda x: int(x, 2))

    # Rearrange values from the 'counts_simulation' dictionary based on the sorted keys:
    sorted_values_simulation = [counts_simulation[key] for key in sorted_keys]

    # Rearrange values from the 'counts_experiment' dictionary based on the sorted keys:
    sorted_values_experiment = [counts_experiment[key] for key in sorted_keys]

    # Create a figure and a primary y-axis:
    _, axis_1 = plt.subplots(figsize=(8, 6))

    # Calculate the horizontal positions of the bars and set the xticks positions:
    bar_positions_main = np.arange(len(sorted_keys))
    plt.xticks(bar_positions_main, sorted_keys)

    # Define the width of each bar:
    bar_width = 0.2

    # Plot the data from the 'sorted_values_simulation' as bars on the primary y-axis:
    axis_1.bar(
        bar_positions_main - bar_width,
        sorted_values_simulation,
        width=bar_width,
        color="r",
        label="Simulation counts\n(simulator)",
    )

    # Plot the data from the 'sorted_values_experiment' as bars on the primary y-axis:
    axis_1.bar(
        bar_positions_main,
        sorted_values_experiment,
        width=bar_width,
        color="orange",
        label="Experiment counts\n(quantum computer)",
    )

    # Create a secondary y-axis that shares the same x-axis:
    axis_2 = axis_1.twinx()

    # Plot the expected probabilities data as bars on the secondary y-axis:
    axis_2.bar(
        bar_positions_main + bar_width,
        state_probabilities_expected,
        width=bar_width,
        color="green",
        label="Expected probabilities\n(input vector)",
        align="center",
    )

    # Find the maximum values of the simulation counts, experiment counts
    # and expected probabilities (the simulation counts and experiment counts
    # need to be normalized to compare them to the probabilities):
    max_counts_simulation = np.max(
        np.divide(np.asarray(sorted_values_simulation, dtype=np.float64), shots)
    )
    max_counts_experiment = np.max(
        np.divide(np.asarray(sorted_values_experiment, dtype=np.float64), shots)
    )
    max_expected = np.max(state_probabilities_expected)

    # Find the maximum value out of the three maximum
    # values above and add 0.1 as a padding for the y-axes.
    # This will be used to align the height of the
    # secondary y-axis with the height of the primary y-axis.
    max = np.max([max_counts_simulation, max_counts_experiment, max_expected]) + 0.1

    # Set the range of the y-axes as the maximum value defined above.
    # The range of the primary y-axis needs to be
    # scaled by the number of shots of the simulation.
    axis_1.set_ylim(0, max * shots)
    axis_2.set_ylim(0, max)

    # Add a title and labels, and show the plot:
    axis_1.set_xlabel("State")
    axis_1.set_ylabel("Count", color="r")
    axis_2.set_ylabel("Probability", color="g")
    axis_1.legend(loc="upper left")
    axis_2.legend(loc="upper right")
    plt.title("Expectations and results")
    plt.show()


# ------------------------------------------------------------------------------
def calculate_fidelity(
    shots: int, statevector: list[Number], counts_experiment: Counts
) -> float:
    """
    Calculates the fidelity between two pure quantum states.
    The 'statevector' parameter represents the first state.
    The second state is obtained by approximation from the
    quantum computer experiment results stored in the
    'counts_experiment' parameter.

    The function first extracts the keys (state labels) from the
    'counts_experiment' dictionary and sorts them based on the
    least significant bit. It then rearranges the values from the
    'counts_experiment' dictionary based on the sorted keys. The sorted
    values are then divided by the number of shots and squared. The
    function then calculates the inner product of the 'statevector' and
    the sorted values. Absolute values are used when calculating the
    inner product. Finally, the function calculates and returns the
    fidelity by squaring the absolute value of the inner product.

    Parameters:
    -----------
    shots: int
        The number of shots used when running
        the circuit on a quantum computer.
    statevector: list[numbers.Number]
        The statevector representing the quantum state.
    counts_experiment: qiskit.result.counts.Counts
        The counts dictionary containing the measurement
        results obtained from a quantum computer experiment.

    Returns:
    --------
    float
        The fidelity between two quantum states, one of
        which is obtained by approximation from quantum
        computer experiment results.
    """

    # Extract the keys (state labels) from
    # the 'counts_experiment' dictionary:
    keys_counts = list(counts_experiment.keys())

    # Sort the keys based on the least significant bit:
    sorted_keys = sorted(keys_counts, key=lambda x: int(x, 2))

    # Rearrange values from the 'counts_experiment'
    # dictionary based on the sorted keys:
    sorted_values = [counts_experiment[key] for key in sorted_keys]

    # Divide the sorted values by the number of 'shots' (this
    # normalizes the measurement outcomes to allow them
    # to be interpreted as probabilities):
    probabilities_experiment = np.divide(sorted_values, shots)

    # Square the probabilities to create a statevector of a
    # hypothetical quantum state (the resulting statevector
    # may not correspond to the quantum computer state;
    # this is only an approximation):
    statevector_experiment = np.sqrt(probabilities_experiment)

    # Compute the inner product:
    inner_product = sum(
        # Multiply the elements in the lists (the absolute values of
        # the 'statevector' elements must be used because the
        # 'statevector_experiment' list only contains positive numbers):
        np.abs(psi_conj) * phi_i
        # Create complex conjugates of the 'statevector' elements
        # and leave the 'statevector_experiment' unchanged:
        for psi_conj, phi_i in zip(statevector, statevector_experiment)
    )

    # Calculate the fidelity and return it:
    return abs(inner_product) ** 2


# ------------------------------------------------------------------------------
def calculate_trace_distance(
    shots: int, statevector: list[Number], counts_experiment: Counts
) -> float:
    """
    Calculates the trace distance between two pure quantum
    states. The 'statevector' parameter represents the first
    state. The second state is obtained by approximation
    from the quantum computer experiment results stored
    in the 'counts_experiment' parameter.

    The function first calls the 'calculate_fidelity'
    function to get the fidelity between the two states
    described above. It then calculates and returns the
    trace distance between those two states by
    taking a square root of '1 - fidelity'.

    Parameters:
    -----------
    shots: int
        The number of shots used when running
        the circuit on a quantum computer.
    statevector: list[numbers.Number]
        The statevector representing the quantum state.
    counts_experiment: qiskit.result.counts.Counts
        The counts dictionary containing the measurement
        results obtained from a quantum computer experiment.

    Returns:
    --------
    float
        The trace distance between two quantum states, one of
        which is obtained by approximation from quantum
        computer experiment results.
    """

    # Fidelity can be used to calculate the trace distance between
    # two pure quantum states. Call the 'calculate_fidelity' function
    # to calculate the fidelity between the state represented by the
    # 'statevector' parameter and the state obtained by approximation
    # from the quantum computer experiment results stored
    # in the 'counts_experiment' parameter:
    fidelity = calculate_fidelity(shots, statevector, counts_experiment)

    # Calculate the trace distance by taking a
    # square root of '1 - fidelity' and return it:
    return np.sqrt(1 - fidelity)
