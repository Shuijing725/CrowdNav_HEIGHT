import gym
import numpy as np
from numpy.linalg import norm
import copy
import os
from collections import defaultdict
from PIL import Image, ImageFilter
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

import pybullet as p
from pybullet_utils import bullet_client

def draw_combined_scene_with_attn(
        pb_client,
        camera_w,
        camera_h,
        view_matrix,
        projection_matrix,
        render_fov,
        robot_uid,
        human_uids,       
        attn_weights_rh,  
        attn_weights_hh, 
        threshold=0.01, 
        crop_x1_ratio=322/900,
        crop_y1_ratio=322/900,
        crop_x2_ratio=600/900,
        crop_y2_ratio=600/900,
        final_w=900,      # after resize
        final_h=900,
        save_path="combined.png"
    ):
        """
        1) Gets the raw PyBullet camera image of size (camera_w, camera_h).
        2) Crops [x1, x2)*[y1, y2), resizes to (final_w, final_h).
        3) Projects 3D points -> raw 2D coords, adjusts them for crop & resize.
        4) Overlays curly arrows for attention, ignoring any attn <= threshold.
        5) Saves the combined figure.

        Args:
        pb_client: your bullet_client or p
        camera_w, camera_h: raw image dims from getCameraImage(...)
        view_matrix, projection_matrix: from your code
        render_fov: for building the same projection matrix if needed
        robot_uid: PyBullet UID for the robot
        human_uids: list of UIDs in the same order as attn arrays
        attn_weights_rh: shape=[N] (1D) for robot->human
        attn_weights_hh: shape=[N,N] or [1,N,N] for human->human
        threshold: skip lines <= threshold
        crop_x1_ratio, crop_y1_ratio, etc: crop ratios from your code
        final_w, final_h: final image size after resizing
        save_path: where to save the result
        """

        # --------------------------------------------------
        # 1) Get the raw image from PyBullet
        # --------------------------------------------------
        raw_image_data = pb_client.getCameraImage(
            width=camera_w,
            height=camera_h,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pb_client.ER_TINY_RENDERER
        )
        # raw_image_data[2] is the RGBA buffer
        raw_rgba = np.reshape(raw_image_data[2], (camera_h, camera_w, 4))
        raw_rgb = raw_rgba[..., :3]

        # --------------------------------------------------
        # 2) Crop & resize the image the way your code does
        # --------------------------------------------------
        # Coordinates in the raw image:
        x1 = int(crop_x1_ratio * camera_w)
        y1 = int(crop_y1_ratio * camera_h)
        x2 = int(crop_x2_ratio * camera_w)
        y2 = int(crop_y2_ratio * camera_h)


        pil_im = Image.fromarray(raw_rgb)
        top = camera_h - y2
        left = x1
        right = x2
        bottom = camera_h - y1

        # If top>bottom, swap them if needed
        if top > bottom:
            top, bottom = bottom, top

        pil_im = pil_im.crop((left, top, right, bottom))

        # Now "pil_im" is the cropped region
        pil_im = pil_im.resize((final_w, final_h), Image.LANCZOS)
        pil_im = pil_im.filter(ImageFilter.SHARPEN)
        final_img = np.array(pil_im)

        # The final displayed image is (final_h, final_w, 3).

        def project_point_3d_to_raw_uv(px, py, pz):
            """
            Returns (u_raw, v_raw) in the un-cropped image coordinate system,
            with (0,0) at the top-left or bottom-left depending on PyBullet config.
            We'll see how to handle the final transform below.
            """
            vm = np.array(view_matrix).reshape((4,4), order='F')
            pm = np.array(projection_matrix).reshape((4,4), order='F')
            vec = np.array([px, py, pz, 1.0], dtype=np.float32)

            clip = pm @ (vm @ vec)
            w = clip[3]
            if abs(w) < 1e-6:
                return None

            ndc_x = clip[0] / w
            ndc_y = clip[1] / w
            ndc_z = clip[2] / w
            # skip if out of [-1,1]
            if not (-1 <= ndc_x <= 1 and -1 <= ndc_y <= 1 and 0 <= ndc_z <= 1):
                return None

            # PyBullet often puts (0,0) at the lower-left corner:
            u = int((ndc_x * 0.5 + 0.5) * camera_w)
            v_bottom = int((ndc_y * 0.5 + 0.5) * camera_h)
            v = camera_h - v_bottom
            return (u, v)

        # --------------------------------------------------
        # 4) Gather positions
        # --------------------------------------------------
        robot_pos, _ = pb_client.getBasePositionAndOrientation(robot_uid)
        # e.g. robot_pos = (rx, ry, rz)
        # Humans
        humans_pos = {}
        for uid in human_uids:
            humans_pos[uid], _ = pb_client.getBasePositionAndOrientation(uid)

        # --------------------------------------------------
        # 5) For each 3D point, get raw (u_raw, v_raw)
        #    Then map it into final coords
        # --------------------------------------------------
        def raw_uv_to_final_uv(u_raw, v_raw):
            """
            Takes a point in the raw image coordinate system,
            checks if it lies in the crop, then transforms it
            into the final (resized) coordinate system.
            Returns (u_final, v_final) or None if out of crop.
            """
            if u_raw is None or v_raw is None:
                return None
            if not (left <= u_raw < right and top <= v_raw < bottom):
                return None

            # Shift inside the cropped region
            u_cropped = u_raw - left
            v_cropped = v_raw - top

            scale_x = final_w / (right - left)
            scale_y = final_h / (bottom - top)

            u_final = u_cropped * scale_x
            v_final = v_cropped * scale_y
            return (u_final, v_final)

        robot_uv_final = None
        # Project the robot's position if you like a slight offset in Z:
        r3d = (robot_pos[0], robot_pos[1], robot_pos[2])  # or robot_pos[2]+0.2 to place arrow above
        r_uv_raw = project_point_3d_to_raw_uv(*r3d)
        if r_uv_raw is not None:
            robot_uv_final = raw_uv_to_final_uv(*r_uv_raw)

        humans_uv_final = {}
        for uid in human_uids:
            hx, hy, hz = humans_pos[uid]
            h_uv_raw = project_point_3d_to_raw_uv(hx, hy, hz)
            if h_uv_raw is not None:
                humans_uv_final[uid] = raw_uv_to_final_uv(*h_uv_raw)
            else:
                humans_uv_final[uid] = None

        # --------------------------------------------------
        # 6) Make a figure, overlay curly arrows
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.imshow(final_img)
        ax.axis('off')

        cmap = plt.cm.viridis
        norm = Normalize(vmin=0.0, vmax=1.0)

        if attn_weights_rh is not None and robot_uv_final is not None:
            arr_rh = attn_weights_rh.squeeze().cpu().detach().numpy()  # shape [N]
            for i, uid in enumerate(human_uids):
                val = arr_rh[i]
                if val < 0.01:
                    continue  # skip meaningless attention
                huv = humans_uv_final[uid]
                if huv is None:
                    continue  # off-screen or out-of-crop
                (xr, yr) = robot_uv_final
                (xh, yh) = huv
                color_line = cmap(norm(val))
                arrow = FancyArrowPatch(
                    posA=(xr, yr),
                    posB=(xh, yh),
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.15',
                    color=color_line,
                    mutation_scale=30,
                    linewidth=2
                )
                ax.add_patch(arrow)
                # Label near midpoint
                # mx, my = 0.5*(xr + xh), 0.5*(yr + yh)
                # ax.text(mx, my, f"{val:.2f}", color=color_line,
                #         fontsize=10, fontweight='bold', ha='center', va='bottom')

        if attn_weights_hh is not None:
            if attn_weights_hh.ndim == 3:
                hh_matrix = attn_weights_hh[0].cpu().detach().numpy()
            else:
                hh_matrix = attn_weights_hh.cpu().detach().numpy()
            n = hh_matrix.shape[0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w_ij = hh_matrix[i,j]
                    w_ji = hh_matrix[j,i]
                    # skip if both are below threshold
                    if (w_ij < 0.01) or (w_ji < 0.01):
                        continue
                    if w_ij <= 0.1:
                        continue
                    uid_i = human_uids[i]
                    uid_j = human_uids[j]

                    pt_i = humans_uv_final[uid_i]
                    pt_j = humans_uv_final[uid_j]
                    if pt_i is None or pt_j is None:
                        # off-screen or out-of-crop => skip
                        continue

                    (xi, yi) = pt_i
                    (xj, yj) = pt_j
                    arrow_color = cmap(norm(w_ij))

                    # Draw arrow i->j
                    arrow_ij = FancyArrowPatch(
                        posA=(xi, yi),
                        posB=(xj, yj),
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.2',  # curly arc
                        color=arrow_color,
                        mutation_scale=30,
                        linewidth=2
                    )
                    ax.add_patch(arrow_ij)

                    # Label near the midpoint, offset slightly so forward/backward arrows don't overlap
                    mx = 0.5 * (xi + xj)
                    my = 0.5 * (yi + yj)

                    # Vector from i->j
                    dx = xj - xi
                    dy = yj - yi
                    length = max(1e-6, np.hypot(dx, dy))

                    # Unit vector along arrow (ax, ay)
                    ax_vec = dx / length
                    ay_vec = dy / length
                    # Unit vector perpendicular (nx, ny)
                    nx = -ay_vec
                    ny =  ax_vec

                    # Decide sign for offset so i->j / j->i end up on different sides
                    sign = 1.0 if i < j else -1.0

                    # Adjust these offsets to taste
                    offset_perp = 10.0    # how many pixels to shift perpendicular
                    offset_along = 0.10* length  # fraction of arrow length to shift along

                    # Final text position
                    # text_x = mx + sign * offset_perp * nx + offset_along * ax_vec
                    # text_y = my + sign * offset_perp * ny + offset_along * ay_vec
                    # ax.text(
                    #     text_x,
                    #     text_y,
                    #     f"{w_ij:.2f}",
                    #     color=arrow_color,
                    #     fontsize=10,
                    #     fontweight='bold',
                    #     ha='center',
                    #     va='center'
                    # )

        # Colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("Attention Weight")

        # save
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)