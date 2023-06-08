% name: exercise_anime.m
% description: stick model doing exercise based on joint kinematics
% author: Vu Phan
% date: 2022/12/16

clear all
close all
clc

% % --- Constants --- %
% % PATH = 'parsed_joint_angles\SUB18';
% PATH = 'parsed_MoCap_kinematics\SUB18';
% 
% % --- Main --- %
% list_exercise_files = dir(PATH);
% flag                = [list_exercise_files.isdir];
% list_exercises      = list_exercise_files(flag);
% exercise_type       = {list_exercises(3:end).name};
% 

ID_LEFTFOOT_COM     = 2:4;
ID_LEFTSHANK_COM    = 8:10;
ID_LEFTTHIGH_COM    = 14:16;
ID_PELVIS_COM       = 20:22;
ID_RIGHTFOOT_COM    = 26:28;
ID_RIGHTSHANK_COM   = 32:34;
ID_RIGHTTHIGH_COM   = 38:40;

ID_LEFTFOOT_ORIENT     = 5:7;
ID_LEFTSHANK_ORIENT    = 11:13;
ID_LEFTTHIGH_ORIENT    = 17:19;
ID_PELVIS_ORIENT       = 23:25;
ID_RIGHTFOOT_ORIENT    = 29:31;
ID_RIGHTSHANK_ORIENT   = 35:37;
ID_RIGHTTHIGH_ORIENT   = 41:43;

DIR_X = 1;
DIR_Y = 2;
DIR_Z = 3;

COLOR_GRAY = [.7 .7 .7];
COLOR_BLACK = 'k';

CAM_ANGLE = -20;

% Walk_01_rep1 SqDL_01_rep3 SqDL_01_rep4 DeclineSq_01_rep1 FwJump_01_rep1 Run_01_rep3

fn = 'HeelRaise_01_rep1.csv';
dt = readmatrix(fn);
dt = dt(4:end, :);

O = zeros(3, 1);
frame_ax = [0.1; 0; 0];
frame_ay = [0; 0.1; 0];
frame_az = [0; 0; 0.1];

% COM position
leftfoot_com    = dt(:, ID_LEFTFOOT_COM);
leftshank_com   = dt(:, ID_LEFTSHANK_COM);
leftthigh_com   = dt(:, ID_LEFTTHIGH_COM);
pelvis_com      = dt(:, ID_PELVIS_COM);
rightfoot_com   = dt(:, ID_RIGHTFOOT_COM);
rightshank_com  = dt(:, ID_RIGHTSHANK_COM);
rightthigh_com  = dt(:, ID_RIGHTTHIGH_COM);

% Orientation
leftfoot_orient    = dt(:, ID_LEFTFOOT_ORIENT);
leftshank_orient   = dt(:, ID_LEFTSHANK_ORIENT);
leftthigh_orient   = dt(:, ID_LEFTTHIGH_ORIENT);
pelvis_orient      = dt(:, ID_PELVIS_ORIENT);
rightfoot_orient   = dt(:, ID_RIGHTFOOT_ORIENT);
rightshank_orient  = dt(:, ID_RIGHTSHANK_ORIENT);
rightthigh_orient  = dt(:, ID_RIGHTTHIGH_ORIENT);

num_samples = size(dt, 1);

right_thigh_length = zeros(0, num_samples);

% Video
animeVideo = VideoWriter('animeVideo');
animeVideo.FrameRate = 24;
open(animeVideo);

for i = 1:1:num_samples
%     right_thigh_length(i) = norm(rightthigh_com(i, :) - rightshank_com(i, :));    

    scatter3(pelvis_com(i, DIR_Z), ...
        pelvis_com(i, DIR_X), ...
        pelvis_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    hold on
    scatter3(leftfoot_com(i, DIR_Z), ...
        leftfoot_com(i, DIR_X), ...
        leftfoot_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    scatter3(leftshank_com(i, DIR_Z), ...
        leftshank_com(i, DIR_X), ...
        leftshank_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    scatter3(leftthigh_com(i, DIR_Z), ...
        leftthigh_com(i, DIR_X), ...
        leftthigh_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    scatter3(rightfoot_com(i, DIR_Z), ...
        rightfoot_com(i, DIR_X), ...
        rightfoot_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    scatter3(rightshank_com(i, DIR_Z), ...
        rightshank_com(i, DIR_X), ...
        rightshank_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    scatter3(rightthigh_com(i, DIR_Z), ...
        rightthigh_com(i, DIR_X), ...
        rightthigh_com(i, DIR_Y), ...
        'o', 'SizeData', 25, 'MarkerEdgeColor', COLOR_BLACK, 'MarkerFaceColor', COLOR_GRAY);
    
    plot3([leftthigh_com(i, DIR_Z), rightthigh_com(i, DIR_Z)], ...
        [leftthigh_com(i, DIR_X), rightthigh_com(i, DIR_X)], ...
        [leftthigh_com(i, DIR_Y), rightthigh_com(i, DIR_Y)], ...
        'Color', COLOR_BLACK, 'LineWidth', 0.8, 'LineStyle', '--')
    plot3([leftthigh_com(i, DIR_Z), leftshank_com(i, DIR_Z)], ...
        [leftthigh_com(i, DIR_X), leftshank_com(i, DIR_X)], ...
        [leftthigh_com(i, DIR_Y), leftshank_com(i, DIR_Y)], ...
        'Color', COLOR_BLACK, 'LineWidth', 0.8, 'LineStyle', '--')
    plot3([rightthigh_com(i, DIR_Z), rightshank_com(i, DIR_Z)], ...
        [rightthigh_com(i, DIR_X), rightshank_com(i, DIR_X)], ...
        [rightthigh_com(i, DIR_Y), rightshank_com(i, DIR_Y)], ...
        'Color', COLOR_BLACK, 'LineWidth', 0.8, 'LineStyle', '--')
    plot3([rightshank_com(i, DIR_Z), rightfoot_com(i, DIR_Z)], ...
        [rightshank_com(i, DIR_X), rightfoot_com(i, DIR_X)], ...
        [rightshank_com(i, DIR_Y), rightfoot_com(i, DIR_Y)], ...
        'Color', COLOR_BLACK, 'LineWidth', 0.8, 'LineStyle', '--')
    plot3([leftshank_com(i, DIR_Z), leftfoot_com(i, DIR_Z)], ...
        [leftshank_com(i, DIR_X), leftfoot_com(i, DIR_X)], ...
        [leftshank_com(i, DIR_Y), leftfoot_com(i, DIR_Y)], ...
        'Color', COLOR_BLACK, 'LineWidth', 0.8, 'LineStyle', '--')
    
    xlim([-1, 1])
    ylim([-1.4, 1.4])
    zlim([0, 1.4])
    camorbit(CAM_ANGLE, 0, 'data', [0 0 1]);

    % Original coordinate
    quiver3(O, O, O, frame_ax, O, O, 'Linewidth', 1.2, 'Color', 'b')
    quiver3(O, O, O, O, frame_ay, O, 'Linewidth', 1.2, 'Color', 'r')
    quiver3(O, O, O, O, O, frame_az, 'Linewidth', 1.2, 'Color', 'g')
    text(frame_ax(1), frame_ax(2), frame_ax(3), 'Z', 'Color', 'b')
    text(frame_ay(1)-0.1, frame_ay(2), frame_ay(3), 'X', 'Color', 'r')
    text(frame_az(1), frame_az(2), frame_az(3), 'Y', 'Color', 'g')

    % --- Plot coordinate --- %
    len = 5;
    len2 = 3;
    len3 = 0.5;
    % Pelvis coordinate
    R_pelvis = get_rot_mat(pelvis_orient(i, :));
    [frame_px, frame_py, frame_pz] = get_trans(R_pelvis, frame_ax, frame_ay, frame_az, pelvis_com(i, :));
    plot_segment_ray(pelvis_com(i, :), frame_pz, len);
    plot_coordinate(pelvis_com(i, :), frame_px, frame_py, frame_pz);  
    
    % Right thigh
    R_rightthigh = get_rot_mat(rightthigh_orient(i, :));
    [frame_rtx, frame_rty, frame_rtz] = get_trans(R_rightthigh, frame_ax, frame_ay, frame_az, rightthigh_com(i, :));
    plot_segment_ray(rightthigh_com(i, :), frame_rtz, len2);
    plot_coordinate(rightthigh_com(i, :), frame_rtx, frame_rty, frame_rtz);  
%     text(rightthigh_com(i, 3), rightthigh_com(i, 1), rightthigh_com(i, 2), 'Right', 'Color', 'b', 'HorizontalAlignment', 'right')
    
    % Right shank
    R_rightshank = get_rot_mat(rightshank_orient(i, :));
    [frame_rsx, frame_rsy, frame_rsz] = get_trans(R_rightshank, frame_ax, frame_ay, frame_az, rightshank_com(i, :));
    plot_segment_ray(rightshank_com(i, :), frame_rsz, len2);
    plot_coordinate(rightshank_com(i, :), frame_rsx, frame_rsy, frame_rsz);
    
    % Right foot
    R_rightfoot = get_rot_mat(rightfoot_orient(i, :));
    [frame_rfx, frame_rfy, frame_rfz] = get_trans(R_rightfoot, frame_ax, frame_ay, frame_az, rightfoot_com(i, :));
    plot_segment_ray(rightfoot_com(i, :), frame_rfz, len3);
    plot_coordinate(rightfoot_com(i, :), frame_rfx, frame_rfy, frame_rfz); 
    
    % Left thigh
    R_leftthigh = get_rot_mat(leftthigh_orient(i, :));
    [frame_ltx, frame_lty, frame_ltz] = get_trans(R_leftthigh, frame_ax, frame_ay, frame_az, leftthigh_com(i, :));
    plot_segment_ray(leftthigh_com(i, :), frame_ltz, len2);
    plot_coordinate(leftthigh_com(i, :), frame_ltx, frame_lty, frame_ltz);  
    
    % Left shank
    R_leftshank = get_rot_mat(leftshank_orient(i, :));
    [frame_lsx, frame_lsy, frame_lsz] = get_trans(R_leftshank, frame_ax, frame_ay, frame_az, leftshank_com(i, :));
    plot_segment_ray(leftshank_com(i, :), frame_lsz, len2);
    plot_coordinate(leftshank_com(i, :), frame_lsx, frame_lsy, frame_lsz);
    
    % Left foot
    R_leftfoot = get_rot_mat(leftfoot_orient(i, :));
    [frame_lfx, frame_lfy, frame_lfz] = get_trans(R_leftfoot, frame_ax, frame_ay, frame_az, leftfoot_com(i, :));
    plot_segment_ray(leftfoot_com(i, :), frame_lfz, len3);
    plot_coordinate(leftfoot_com(i, :), frame_lfx, frame_lfy, frame_lfz); 
       
    xlabel('z axis (m)')
    ylabel('x axis (m)')
    zlabel('y axis (m)')

%     drawnow limitrate
    grid off
    pause(0.01)
    frame = getframe(gcf);
    writeVideo(animeVideo, frame);
    clf('reset')
end

close(animeVideo);


% plot(right_thigh_length)

%% Utils
% Get rotation matrix
function [R] = get_rot_mat(xyz_seq)
    a = xyz_seq(1);
    b = xyz_seq(2);
    c = xyz_seq(3);

    Rx = [1 0 0; 0 cosd(a) -sind(a); 0 sind(a) cosd(a)];
    Ry = [cosd(b) 0 sind(b); 0 1 0; -sind(b) 0 cosd(b)];
    Rz = [cosd(c) -sind(c) 0; sind(c) cosd(c) 0; 0 0 1];

    R = Rx*Ry*Rz;
end

% Obtain transformation
function [frame_px, frame_py, frame_pz] = get_trans(R, frame_ax, frame_ay, frame_az, com)
    T = [R com'; 0 0 0 1];
    frame_px = T*[frame_ax; 1];
    frame_px = frame_px(1:3);
    frame_py = T*[frame_ay; 1];
    frame_py = frame_py(1:3);
    frame_pz = T*[frame_az; 1];
    frame_pz = frame_pz(1:3);
end

% Plot
function plot_coordinate(com, frame_x, frame_y, frame_z)
    DIR_X = 1;
    DIR_Y = 2;
    DIR_Z = 3;

    quiver3(com(DIR_Z), com(DIR_X), com(DIR_Y), ...
        frame_x(DIR_Z) - com(DIR_Z), ...
        frame_x(DIR_X) - com(DIR_X), ...
        frame_x(DIR_Y) - com(DIR_Y), 1.2, 'Color', 'r')
    quiver3(com(DIR_Z), com(DIR_X), com(DIR_Y), ...
        frame_z(DIR_Z) - com(DIR_Z), ...
        frame_z(DIR_X) - com(DIR_X), ...
        frame_z(DIR_Y) - com(DIR_Y), 1.2, 'Color', 'b')
    quiver3(com(DIR_Z), com(DIR_X), com(DIR_Y), ...
        frame_y(DIR_Z) - com(DIR_Z), ...
        frame_y(DIR_X) - com(DIR_X), ...
        frame_y(DIR_Y) - com(DIR_Y), 1.2, 'Color', 'g')
end

% Plot a line through 2 points
function plot_segment_ray(p1, p2, len)

    COLOR_BLACK = 'k';
    COLOR_GRAY = [.7 .7 .7];

    x1 = p1(1);
    y1 = p1(2);
    z1 = p1(3);
    
    x2 = p2(1);
    y2 = p2(2);
    z2 = p2(3);

    nx = x2 - x1;
    ny = y2 - y1;
    nz = z2 - z1;

    xx = [x1 - len*nx, x2 + len*nx];
    yy = [y1 - len*ny, y2 + len*ny];
    zz = [z1 - len*nz, z2 + len*nz];

    plot3(zz, xx, yy, 'Color', COLOR_GRAY, 'LineWidth', 0.5);
end


