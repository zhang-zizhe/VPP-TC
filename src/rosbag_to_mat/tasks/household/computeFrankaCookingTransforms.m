function [H_left_bowl, H_center_bowl, H_right_bowl] = computeFrankaCookingTransforms()

H_left_bowl= eye(4); H_left_bowl(1:3,4) = [0.65, -0.325, 0.15];
H_center_bowl= eye(4); H_center_bowl(1:3,4) = [0.65, -0.05, 0.15];
H_right_bowl= eye(4); H_right_bowl(1:3,4) = [0.65, 0.225, 0.15];

end

