clc; clear all; close all;
%%
pyenv('Version', '/opt/anaconda3/bin/python');
run eidors-v3.11/eidors/startup.m;
eidors_cache('clear_all');
%% Importing mesh to MATLAB

load('electrode_node_labels.mat');
num_electrodes = size(electrode_node_labels, 2);
middle_ix = ceil(num_electrodes / 2);
mid_electrode_label = electrode_node_labels(middle_ix);
%%
fname = "/Users/piotrblaszyk/Downloads/mesh-3.vtu";
fname2 = "foo.mat";

if exist('meshio')
    disp("meshio branch");
    MeshioIsHere = 1;
    M = meshio.read(fname);
    M.cells(1).tri = M.cells(1).tri(:, 1:4);
    save(fname2, 'M');
    meshio.plot(M);
else
    disp("non-meshio branch");
    v=1;
    load(fname);
end
%% Creating the model in Eidors

MDL = eidors_obj('fwd_model' , 'softActuator');
MDL.nodes = M.vtx;
tetrahedra = M.cells(1).tri;
MDL.elems = tetrahedra;
[srf idx] = find_boundary(tetrahedra);
MDL.boundary = srf;
MDL.gnd_node = mid_electrode_label;
%% Plot mesh to check all is ok

show_fem(MDL);
title('Eidors is ok with dimensions in meters');
%% Electrodes

z_contact = 0.0000001;
for i = 1:num_electrodes
    elecs(i).nodes = electrode_node_labels(i);
    elecs(i).z_contact = z_contact;
end
MDL.electrode = elecs;
figure;
show_fem(MDL);
title('Mesh with electrode locations');
MDL = remove_unused_nodes(MDL);
%% Completing Forward model

MDL.solve = @fwd_solve_1st_order;
MDL.jacobian = @jacobian_adjoint;
MDL.system_mat = @system_mat_1st_order;
MDL.normalize_measurements = 0;

Amp = 1;
prt = [
    6, 3, 5, 4;
    5, 2, 4, 3;
    4, 1, 3, 2;
    3, 8, 2, 1;
    2, 9, 1, 8;
    1, 10, 8, 9;
    8, 11, 9, 10;
    9, 12, 10, 11;
    10, 13, 11, 12;
    6, 1, 4, 3;
    1, 13, 9, 11;
    6, 13, 2, 9;
];
stim = stim_meas_list(prt, num_electrodes, Amp);
MDL.stimulation = stim;
%%
% EIT measurements are defined in the following way:
% CS+ CS- V+ V-
% 1,2,1,6
% i.e. inject between 1 and 2, measure 1 referenced to 6 etc.
%% Checking if fwd model is working

if valid_fwd_model(MDL)
    disp('Forward model works with MATLAB');
end
%% Get voltages

S = 1.6;
img = mk_image(MDL,S);
img.fwd_solve.get_all_meas = 1;
foo = fwd_solve(img);
res = foo.meas;