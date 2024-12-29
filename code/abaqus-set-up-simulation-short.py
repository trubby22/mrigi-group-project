from abaqus import *
from abaqusConstants import *
import numpy as np

# amplitudes = np.linspace(0.1, 1.0, 10)
# np.random.shuffle(amplitudes)
amplitudes = np.array([1.0])
period_seconds_0_max = 1.0
timesteps_0_max = 100
timestep_seconds = period_seconds_0_max / timesteps_0_max
timesteps_0_max_0 = timesteps_0_max * 2
timesteps_float = amplitudes * timesteps_0_max_0
timesteps = np.ceil(timesteps_float).astype(int)
num_total_timesteps = np.sum(timesteps)
time_points = np.zeros(num_total_timesteps)
amplitude_values = np.zeros(num_total_timesteps)

timestep = 0
for i in range(len(amplitudes)):
    amplitude = amplitudes[i]
    sequence_timesteps = timesteps[i]
    timestep_sequence = np.arange(sequence_timesteps)
    amplitude_values_sequence = amplitude * (0.5 - 0.5 * np.cos(2 * np.pi * 1.0 / (sequence_timesteps - 1) * timestep_sequence))
    timestep_sequence_seconds = (timestep_sequence + timestep) * timestep_seconds
    time_points[timestep : timestep + sequence_timesteps] = timestep_sequence_seconds
    amplitude_values[timestep : timestep + sequence_timesteps] = amplitude_values_sequence
    timestep += sequence_timesteps

mid = num_total_timesteps / 2 + 1
time_points = time_points[:mid]
amplitude_values = amplitude_values[:mid]
print("num timesteps")
print(mid)
print("amplitudes")
print(amplitudes)
print("timestep")
print(timestep_seconds)
print("total duration")
print(time_points[-1])
amplitude_data = zip(time_points, amplitude_values)
mdb.models['Model-2'].TabularAmplitude(
    name='PythonAmplitude',
    timeSpan=STEP,
    smooth=SOLVER_DEFAULT,
    data=amplitude_data
)

execfile("C:\\Users\\Abaqus\\Documents\\code\\abaqus-set-up-simulation.py")
