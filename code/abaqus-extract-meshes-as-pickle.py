from odbAccess import openOdb
import numpy as np
import pickle

# odb_file_path = "C:\\Users\\Abaqus\\Documents\\abaqus-tutorial\\our-model.odb"
odb_file_path = "C:\\Abaqus_temp\\our-model.odb"
output_file_path = "C:\\Users\\Abaqus\\Documents\\code\\mesh.pkl"

odb = openOdb(path=odb_file_path)
assembly = odb.rootAssembly
steps = odb.steps.items()
_, instance = assembly.instances.items()[0]

conns = [list(elem.connectivity) for elem in instance.elements]
element_types = set([elem.type for elem in instance.elements])
print(element_types)

for i in range(len(conns)):
    conn = conns[i]
    for j in range(len(conn)):
        conns[i][j] -= 1

total_frames = 0
all_coords = []
initial_coords = []
for step_name, step in steps:
    if total_frames == 0:
        for node in instance.nodes:
            initial_coords.append(node.coordinates)
    if step_name == "Press":
        frames = step.frames
        for frame in frames:
            displacement = frame.fieldOutputs['U']
            node_displacements = {disp.nodeLabel: disp.data for disp in displacement.values}
            coords = []
            for node in instance.nodes:
                label = node.label
                displacement = node_displacements.get(label, [0, 0, 0])
                original_coords = node.coordinates
                updated_coords = [original_coords[i] + displacement[i] for i in range(3)]
                coords.append(updated_coords)
            all_coords.append(coords)
            total_frames += 1

odb.close()

assert(len(set([len(x) for x in all_coords])) == 1)

initial_coords_np = np.array(initial_coords)
all_coords_np = np.array(all_coords)
conns_np = np.array(conns)
foo = (initial_coords_np, all_coords_np, conns_np)
with open(output_file_path, "wb") as file:
    pickle.dump(foo, file)

print(total_frames)
