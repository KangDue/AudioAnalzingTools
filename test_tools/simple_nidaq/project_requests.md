Python Program Specification Using nidaqmx for Data Acquisition
1. Device Detection and Selection
The program should automatically detect all NI DAQ (nidaqmx) devices connected to the PC at launch.

Allow the user to select one device from the detected list for further use.

2. Input Channel Configuration (ai0, ai1, ai2, ai3)
The selected device uses input channels fixed at ai0, ai1, ai2, and ai3.

For each channel, allow the user to configure parameters using the appropriate API:

Use add_ai_voltage_chan, add_ai_accel_chan, and add_ai_microphone_chan as base options.

A tab or form UI must allow per-channel parameter editing and selection (including channel type: acceleration, microphone, voltage, etc.).

IEPE-related parameters (e.g., excitation current) must always be included (never omitted) for relevant sensor types.

All other relevant parameters (e.g., sensitivity, units, input range, max sound pressure level) should be user-configurable, with sensible default values pre-filled.

Save user settings locally and automatically reload them next time the program starts.

3. Measurement Configuration (Main Tab)
Main UI should provide controls for:

Sampling rate

Number of samples to read per call

Channel enable/disable (using checkboxes per input channel)

Any other task-level config required for data reading.

4. Real-Time Data Plotting
Live plot the incoming data using PyQtGraph (or comparable library); allow zooming and panning.

The user should be able to adjust how much past data remains visible in the plot (e.g., “plot window length”).

Plotting widgets must allow standard interactions, such as zoom, pan, and rescale.

5. Data Saving
Implement functionality to automatically save measurement data at regular time intervals as set by the user.

Files should be named in the format YYYYMMDD_HHMMSS_duration.npy (timestamp and duration unit).

Clearly indicate the units of measurement stored in the file name or metadata.

Data must be stored in a lossless, portable format (e.g., NumPy .npy).

6. Additional Required Features
Ensure error handling for:

Device disconnection

Channel misconfiguration

File write failures

Include start/stop measurement controls.

Show the currently active configuration in the UI.

Optionally, a simple log/output window for status or error messages.

References for the expected usage and best practices:

Official nidaqmx Python documentation for task/channel setup and parameter options

PyQtGraph’s real-time plotting features and examples

You can share this summary as a spec with any developer familiar with Python, PyQt (or PySide), and NO-DAQmx to begin implementing the application according to your needs.