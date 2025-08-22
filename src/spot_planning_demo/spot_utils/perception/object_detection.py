""""Interface for detecting objects with fiducials or pretrained models.

NOTE: the april tag code is removed for now.

The fiducials are april tags. The size of the april tag is important and can be
configured via CFG.spot_fiducial_size.

The pretrained models are currently DETIC and SAM (used together). DETIC takes
a language description of an object (e.g., "brush") and an RGB image and finds
a bounding box. SAM segments objects within the bounding box (class-agnostic).
The associated depth image is then used to estimate the depth of the object
based on the median depth of segmented points. See the README in this directory
for instructions on setting up DETIC and SAM on a server.

Object detection returns SE3Poses in the world frame but only x, y, z positions
are currently detected. Rotations should be ignored.
"""

import io
import logging
from functools import partial
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import dill as pkl
import numpy as np
import PIL.Image
import requests
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from spot_planning_demo.spot_utils.perception.perception_structs import (
    KnownStaticObjectDetectionID,
    LanguageObjectDetectionID,
    ObjectDetectionID,
    PythonicObjectDetectionID,
    RGBDImage,
    RGBDImageWithContext,
    SegmentedBoundingBox,
)
from spot_planning_demo.spot_utils.utils import get_graph_nav_dir

# Hack to avoid double image capturing when we want to (1) get object states
# and then (2) use the image again for pixel-based grasping.
_LAST_DETECTED_OBJECTS: Tuple[
    Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]
] = ({}, {})


# This is legacy; remove later or add back.
OBJECT_SPECIFIC_GRASP_SELECTORS: dict[ObjectDetectionID, Any] = {}


def get_last_detected_objects() -> (
    Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]
):
    """Return the last output from detect_objects(), ignoring inputs."""
    return _LAST_DETECTED_OBJECTS


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],  # camera name to RGBD
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect object poses (in the world frame!) from RGBD.

    Each object ID is assumed to exist at most once in each image, but can exist in
    multiple images.

    The second return value is a collection of artifacts that can be useful for
    debugging / analysis.
    """
    global _LAST_DETECTED_OBJECTS  # pylint: disable=global-statement

    # Collect and dispatch.
    language_object_ids: Set[LanguageObjectDetectionID] = set()
    pythonic_object_ids: Set[PythonicObjectDetectionID] = set()
    known_static_object_ids: Set[KnownStaticObjectDetectionID] = set()
    for object_id in object_ids:
        if isinstance(object_id, KnownStaticObjectDetectionID):
            known_static_object_ids.add(object_id)
        elif isinstance(object_id, PythonicObjectDetectionID):
            pythonic_object_ids.add(object_id)
        else:
            assert isinstance(object_id, LanguageObjectDetectionID)
            language_object_ids.add(object_id)
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {"language": {}}

    # Read off known objects directly.
    for known_obj_id in known_static_object_ids:
        detections[known_obj_id] = known_obj_id.pose

    # There is batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgbds, allowed_regions
    )
    detections.update(language_detections)
    artifacts["language"] = language_artifacts

    # Handle pythonic object detection.
    for object_id in pythonic_object_ids:
        detection = object_id.fn(rgbds)
        if detection is not None:
            detections[object_id] = detection
            break

    _LAST_DETECTED_OBJECTS = (detections, artifacts)

    return detections, artifacts


def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict]:
    """Detect an object pose using a vision-language model.

    The second return value is a dictionary of "artifacts", which include the raw
    vision-language detection results. These are primarily useful for debugging /
    analysis. See visualize_all_artifacts().
    """

    object_id_to_img_detections = _query_detic_sam(object_ids, rgbds)

    # Convert the image detections into pose detections. Use the best scoring
    # image for which a pose can be successfully extracted.

    def _get_detection_score(
        img_detections: Dict[str, SegmentedBoundingBox], camera: str
    ) -> float:
        return img_detections[camera].score

    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    for obj_id, img_detections in object_id_to_img_detections.items():
        # Consider detections from best (highest) to worst score.
        for camera in sorted(
            img_detections,
            key=partial(_get_detection_score, img_detections),
            reverse=True,
        ):
            seg_bb = img_detections[camera]
            rgbd = rgbds[camera]
            pose = _get_pose_from_segmented_bounding_box(seg_bb, rgbd)
            # Pose extraction can fail due to depth reading issues. See
            # docstring of _get_pose_from_segmented_bounding_box for more.
            if pose is None:
                continue
            # If the detected pose is outside the allowed bounds, skip.
            pose_xy = np.array([pose.x, pose.y])
            if allowed_regions is not None:
                in_allowed_region = False
                for region in allowed_regions:
                    if region.find_simplex(pose_xy).item() >= 0:
                        in_allowed_region = True
                        break
                if not in_allowed_region:
                    logging.info(
                        "WARNING: throwing away detection for "
                        + f"{obj_id} because it's out of bounds. "
                        + f"(pose = {pose_xy})"
                    )
                    continue
            # Pose extraction succeeded.
            detections[obj_id] = pose
            break

    # Save artifacts for analysis and debugging.
    artifacts = {
        "rgbds": rgbds,
        "object_id_to_img_detections": object_id_to_img_detections,
    }

    return detections, artifacts


def rotate_point_in_image(
    r: float, c: float, rot_degrees: float, height: int, width: int
) -> Tuple[int, int]:
    """If an image has been rotated using ndimage.rotate, this computes the location of
    a pixel (r, c) in that image following the same rotation.

    The rotation is expected in degrees, following ndimage.rotate.
    """
    rotation_radians = np.radians(rot_degrees)
    transform_matrix = np.array(
        [
            [np.cos(rotation_radians), -np.sin(rotation_radians)],
            [np.sin(rotation_radians), np.cos(rotation_radians)],
        ]
    )
    # Subtract the center of the image from the pixel location to
    # translate the rotation to the origin.
    center = np.array([(height - 1) / 2.0, (width - 1) / 2])
    centered_pt = np.subtract([r, c], center)
    # Apply the rotation.
    rotated_pt_centered = np.matmul(transform_matrix, centered_pt)
    # Add the center of the image back to the pixel location to
    # translate the rotation back from the origin.
    rotated_pt = rotated_pt_centered + center
    return rotated_pt[0], rotated_pt[1]


def _query_detic_sam(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext] | Dict[str, RGBDImage],
    max_server_retries: int = 5,
    detection_threshold: float = 0.5,
) -> Dict[ObjectDetectionID, Dict[str, SegmentedBoundingBox]]:
    """Returns object ID to image ID (camera) to segmented bounding box."""

    object_id_to_img_detections: Dict[
        ObjectDetectionID, Dict[str, SegmentedBoundingBox]
    ] = {obj_id: {} for obj_id in object_ids}

    # Create buffer dictionary to send to server.
    buf_dict = {}
    for camera_name, rgbd in rgbds.items():
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)  # type: ignore
        buf_dict[camera_name] = _image_to_bytes(pil_rotated_img)

    # Extract all the classes that we want to detect.
    classes = sorted(o.language_id for o in object_ids)

    # Query server, retrying to handle possible wifi issues.
    for _ in range(max_server_retries):
        try:
            r = requests.post(
                "http://localhost:5550/batch_predict",
                files=buf_dict,
                data={"classes": ",".join(classes)},
                timeout=100,
            )
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        logging.warning("DETIC-SAM FAILED, POSSIBLE SERVER/WIFI ISSUE")
        return object_id_to_img_detections

    # If the status code is not 200, then fail.
    if r.status_code != 200:
        logging.warning(f"DETIC-SAM FAILED! STATUS CODE: {r.status_code}")
        return object_id_to_img_detections

    # Querying the server succeeded; unpack the contents.
    with io.BytesIO(r.content) as f:
        try:
            server_results = np.load(f, allow_pickle=True)
        # Corrupted results.
        except pkl.UnpicklingError:
            logging.warning("DETIC-SAM FAILED DURING UNPICKLING!")
            return object_id_to_img_detections

        # Process the results and save all detections per object ID.
        for camera_name, rgbd in rgbds.items():
            rot_boxes = server_results[f"{camera_name}_boxes"]
            ret_classes = server_results[f"{camera_name}_classes"]
            rot_masks = server_results[f"{camera_name}_masks"]
            scores = server_results[f"{camera_name}_scores"]

            # Invert the rotation immediately so we don't need to worry about
            # them henceforth.
            h, w = rgbd.rgb.shape[:2]
            image_rot = rgbd.image_rot
            boxes = [_rotate_bounding_box(bb, -image_rot, h, w) for bb in rot_boxes]
            masks = [
                ndimage.rotate(m.squeeze(), -image_rot, reshape=False)
                for m in rot_masks
            ]

            # Filter out detections by confidence. We threshold detections
            # at a set confidence level minimum, and if there are multiple,
            # we only select the most confident one. This structure makes
            # it easy for us to select multiple detections if that's ever
            # necessary in the future.
            for obj_id in object_ids:
                # If there were no detections (which means all the
                # returned values will be numpy arrays of shape (0, 0))
                # then just skip this source.
                if ret_classes.size == 0:
                    continue
                obj_id_mask = ret_classes == obj_id.language_id
                if not np.any(obj_id_mask):
                    continue
                max_score = np.max(scores[obj_id_mask])
                best_idx = np.where(scores == max_score)[0].item()
                if scores[best_idx] < detection_threshold:
                    continue
                # Save the detection.
                seg_bb = SegmentedBoundingBox(
                    boxes[best_idx], masks[best_idx], scores[best_idx]
                )
                object_id_to_img_detections[obj_id][rgbd.camera_name] = seg_bb

    return object_id_to_img_detections


def _image_to_bytes(img: PIL.Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _rotate_bounding_box(
    bb: Tuple[float, float, float, float], rot_degrees: float, height: int, width: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb
    ry1, rx1 = rotate_point_in_image(y1, x1, rot_degrees, height, width)
    ry2, rx2 = rotate_point_in_image(y2, x2, rot_degrees, height, width)
    return (rx1, ry1, rx2, ry2)


def _get_pose_from_segmented_bounding_box(
    seg_bb: SegmentedBoundingBox, rgbd: RGBDImageWithContext, min_depth_value: float = 2
) -> Optional[math_helpers.SE3Pose]:
    """Returns None if the depth of the object cannot be estimated.

    The known case where this happens is when the robot's hand occludes the depth camera
    (which is physically above the RGB camera).
    """
    # Get the center of the bounding box.
    x1, y1, x2, y2 = seg_bb.bounding_box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Get the median depth value of segmented points.
    # Filter 0 points out of the depth map.
    seg_mask = seg_bb.mask & (rgbd.depth > min_depth_value)
    segmented_depth = rgbd.depth[seg_mask]
    # See docstring.
    if len(segmented_depth) == 0:
        # logging.warning doesn't work here because of poor spot logging.
        print("WARNING: depth reading failed. Is hand occluding?")
        return None
    depth_value = np.median(segmented_depth)

    # Convert to camera frame position.
    fx = rgbd.camera_model.intrinsics.focal_length.x
    fy = rgbd.camera_model.intrinsics.focal_length.y
    cx = rgbd.camera_model.intrinsics.principal_point.x
    cy = rgbd.camera_model.intrinsics.principal_point.y
    depth_scale = rgbd.depth_scale
    camera_z = depth_value / depth_scale
    camera_x = np.multiply(camera_z, (x_center - cx)) / fx
    camera_y = np.multiply(camera_z, (y_center - cy)) / fy
    camera_frame_pose = math_helpers.SE3Pose(
        x=camera_x, y=camera_y, z=camera_z, rot=math_helpers.Quat()
    )

    # Convert camera to world.
    world_frame_pose = rgbd.world_tform_camera * camera_frame_pose

    # The angles are not meaningful, so override them.
    final_pose = math_helpers.SE3Pose(
        x=world_frame_pose.x,
        y=world_frame_pose.y,
        z=world_frame_pose.z,
        rot=math_helpers.Quat(),
    )

    return final_pose


def get_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext],
    artifacts: Dict[str, Any],
    object_id: ObjectDetectionID,
    camera_name: str,
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Select a pixel for grasping in the given camera image."""

    if object_id in OBJECT_SPECIFIC_GRASP_SELECTORS:
        selector = OBJECT_SPECIFIC_GRASP_SELECTORS[object_id]
        return selector(rgbds, artifacts, camera_name, rng)

    pixel = get_random_mask_pixel_from_artifacts(artifacts, object_id, camera_name, rng)
    return (pixel[0], pixel[1]), None


def get_random_mask_pixel_from_artifacts(
    artifacts: Dict[str, Any],
    object_id: ObjectDetectionID,
    camera_name: str,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Extract the pixel in the image corresponding to the center of the object with
    object ID.

    The typical use case is to get the pixel to pass into the grasp controller. This is
    a fairly hacky way to go about this, but since the grasp controller parameterization
    is a special case (most users of object detection shouldn't need to care about the
    pixel), we do this.
    """
    assert isinstance(object_id, LanguageObjectDetectionID)
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[object_id][camera_name]
    except KeyError:
        raise ValueError(f"{object_id} not detected in {camera_name}")

    # Select a random valid pixel from the mask.
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    mask_idx = rng.choice(len(pixels_in_mask))
    pixel_tuple = (pixels_in_mask[1][mask_idx], pixels_in_mask[0][mask_idx])
    # Uncomment to plot the grasp pixel being selected!
    # rgb_img = artifacts["language"]["rgbds"][camera_name].rgb
    # _, axes = plt.subplots()
    # axes.imshow(rgb_img)
    # axes.add_patch(
    #     plt.Rectangle((pixel_tuple[0], pixel_tuple[1]), 5, 5, color='red'))
    # plt.tight_layout()
    # plt.savefig("grasp_pixel.png", dpi=300)
    # plt.close()
    return pixel_tuple


def visualize_all_artifacts(
    artifacts: Dict[str, Any], detections_outfile: Path, no_detections_outfile: Path
) -> None:
    """Analyze the artifacts."""
    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    flat_detections: List[
        Tuple[RGBDImageWithContext, LanguageObjectDetectionID, SegmentedBoundingBox]
    ] = []
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            rgbd = rgbds[camera]
            flat_detections.append((rgbd, obj_id, seg_bb))

    # Visualize in subplots where columns are: rotated RGB, original RGB,
    # original depth, bounding box, mask. Each row is one detection, so if
    # there are multiple detections in a single image, then there will be
    # duplicate first cols.
    fig_scale = 2
    if flat_detections:
        _, axes = plt.subplots(
            len(flat_detections),
            5,
            squeeze=False,
            figsize=(5 * fig_scale, len(flat_detections) * fig_scale),
        )
        plt.suptitle("Detections")
        for i, (rgbd, obj_id, seg_bb) in enumerate(flat_detections):
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap="Greys_r", vmin=0, vmax=10000)

            # Bounding box.
            ax_row[3].imshow(rgbd.rgb)
            box = seg_bb.bounding_box
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax_row[3].add_patch(
                plt.Rectangle(
                    (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=1
                )
            )

            ax_row[4].imshow(seg_bb.mask, cmap="binary_r", vmin=0, vmax=1)

            # Labels.
            abbreviated_name = obj_id.language_id
            max_abbrev_len = 24
            if len(abbreviated_name) > max_abbrev_len:
                abbreviated_name = abbreviated_name[:max_abbrev_len] + "..."
            row_label = "\n".join(
                [
                    abbreviated_name,
                    f"[{rgbd.camera_name}]",
                    f"[score: {seg_bb.score:.2f}]",
                ]
            )
            ax_row[0].set_ylabel(row_label, fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")
                ax_row[3].set_xlabel("Bounding Box")
                ax_row[4].set_xlabel("Mask")

        plt.tight_layout()
        plt.savefig(detections_outfile, dpi=300)
        print(f"Wrote out to {detections_outfile}.")
        plt.close()

    # Visualize all of the images that have no detections.
    all_cameras = set(rgbds)
    cameras_with_detections = {r.camera_name for r, _, _ in flat_detections}
    cameras_without_detections = sorted(all_cameras - cameras_with_detections)

    if cameras_without_detections:
        _, axes = plt.subplots(
            len(cameras_without_detections),
            3,
            squeeze=False,
            figsize=(3 * fig_scale, len(cameras_without_detections) * fig_scale),
        )
        plt.suptitle("Cameras without Detections")
        for i, camera in enumerate(cameras_without_detections):
            rgbd = rgbds[camera]
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap="Greys_r", vmin=0, vmax=10000)

            # Labels.
            ax_row[0].set_ylabel(f"[{rgbd.camera_name}]", fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")

        plt.tight_layout()
        plt.savefig(no_detections_outfile, dpi=300)
        print(f"Wrote out to {no_detections_outfile}.")
        plt.close()


def display_camera_detections(artifacts: Dict[str, Any], axes: plt.Axes) -> None:
    """Plot per-camera detections on the given axes.

    The axes are given as input because we might want to update the same axes
    repeatedly, e.g., during object search.
    """

    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    # Organize detections by camera.
    camera_to_rgbd = {rgbd.camera_name: rgbd for rgbd in rgbds.values()}
    camera_to_detections: Dict[
        str, List[Tuple[LanguageObjectDetectionID, SegmentedBoundingBox]]
    ] = {c: [] for c in camera_to_rgbd}
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            camera_to_detections[camera].append((obj_id, seg_bb))

    # Plot per-camera.
    box_colors = ["green", "red", "blue", "purple", "gold", "brown", "black"]
    camera_order = sorted(camera_to_rgbd)
    assert hasattr(axes, "flat")
    for ax, camera in zip(axes.flat, camera_order):
        ax.clear()
        ax.set_title(camera)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the RGB image.
        rgbd = camera_to_rgbd[camera]
        ax.imshow(rgbd.rotated_rgb)

        for i, (obj_id, seg_bb) in enumerate(camera_to_detections[camera]):

            color = box_colors[i % len(box_colors)]

            # Display the bounding box.
            box = seg_bb.bounding_box
            # Rotate.
            image_rot = rgbd.image_rot
            h, w = rgbd.rgb.shape[:2]
            box = _rotate_bounding_box(box, image_rot, h, w)
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=1
                )
            )
            # Label with the detection and score.
            ax.text(
                -250,  # off to the left side
                50 + 60 * i,
                f"{obj_id.language_id}: {seg_bb.score:.2f}",
                color="white",
                fontsize=12,
                fontweight="bold",
                bbox={"facecolor": color, "edgecolor": color, "alpha": 0.5},
            )


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import argparse

    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from spot_planning_demo.spot_utils.perception.spot_cameras import capture_images
    from spot_planning_demo.spot_utils.spot_localization import SpotLocalizer
    from spot_planning_demo.spot_utils.utils import verify_estop

    TEST_CAMERAS = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "frontright_fisheye_image",
    ]
    TEST_LANGUAGE_DESCRIPTIONS = [
        "stuffed animal tiger toy",
    ]

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        parser = argparse.ArgumentParser(description="Parse the robot's hostname.")
        parser.add_argument(
            "--hostname",
            type=str,
            required=True,
            help="The robot's hostname/ip-address (e.g. 192.168.80.3)",
        )
        parser.add_argument(
            "--map_name",
            type=str,
            required=True,
            help="The name of the map folder to load (sub-folder under graph_nav_maps)",
        )
        args = parser.parse_args()

        hostname = args.hostname
        path = get_graph_nav_dir(args.map_name)

        # First, capture images.
        sdk = create_standard_sdk("SpotCameraTestClient")
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(
            lease_client, must_acquire=True, return_at_exit=True
        )

        assert path.exists()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        rgbds = capture_images(robot, localizer, TEST_CAMERAS)

        # Try detection.
        language_ids: List[ObjectDetectionID] = [
            LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
        ]
        known_static_id: ObjectDetectionID = KnownStaticObjectDetectionID(
            "imaginary_box", math_helpers.SE3Pose(-5, 0, 0, rot=math_helpers.Quat())
        )
        object_ids: List[ObjectDetectionID] = [known_static_id] + language_ids
        detections, artifacts = detect_objects(object_ids, rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile, no_detections_outfile)

    _run_manual_test()
