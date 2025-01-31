import numpy as np
from utils import get_obj_pos, get_obj_names, parse_position, parse_obj_name

# make it bigger by 1.5.
new_shape_pts = scale_pts_around_centroid_np(shape_pts, scale_x=1.5, scale_y=1.5)
# move it to the right by 10cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0.1, 0])
# move it to the top by 20cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0, 0.2])
# rotate it clockwise by 40 degrees.
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=-np.deg2rad(40))
# rotate by 30 degrees and make it slightly smaller
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=np.deg2rad(30))
new_shape_pts = scale_pts_around_centroid_np(new_shape_pts, scale_x=0.7, scale_y=0.7)
# move it toward the blue block.
block_name = parse_obj_name('the blue block', f'objects = {get_obj_names()}')
block_pos = get_obj_pos(block_name)
mean_delta = np.mean(block_pos - shape_pts, axis=1)
new_shape_pts = translate_pts_np(shape_pts, mean_delta)