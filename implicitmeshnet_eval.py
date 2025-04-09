import os
import json

import numpy as np
from numpy.typing import NDArray
import torch
import trimesh
import SimpleITK as sitk
from fire import Fire
from tqdm import tqdm

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, get_vtk_array_type  # pylint: disable=import-error

from shapeestimationnet import ShapeEstimationNet
from implicitmeshnet_train import ImplicitMeshNetTrainConfig, get_mesh_template
from implicitmeshnet_dataset import resample_image, apply_window_level
from gcn import create_edge_index


def write_vtk_polydata(poly: vtk.vtkPolyData, filename: str, verbose: bool = False) -> None:
    """
    This function writes a vtk polydata to disk
    Args:
        poly: vtk polydata
        fn: file name
    Returns:
        None
    """
    if verbose:
        print('Writing vtp with name:', filename)
    if filename == "":
        raise ValueError("filname not set")

    _ , extension = os.path.splitext(filename)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.stl':
        writer = vtk.vtkSTLWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif extension == '.obj':
        writer = vtk.vtkOBJWriter()
    else:
        raise ValueError("Incorrect extension"+extension)
    writer.SetInputData(poly)
    writer.SetFileName(filename)
    writer.Update()
    writer.Write()


def convert_vtk_image_to_numpy(vtk_image: vtk.vtkImageData) -> NDArray:
    """
    Converts a vtkImageData to a numpy array.
    """
    img_array = vtk_image.GetPointData().GetScalars()
    img_array = vtk.util.numpy_support.vtk_to_numpy(img_array)
    img_array = img_array.reshape(vtk_image.GetDimensions(), order='F')
    return img_array


def export_numpy_to_vtk(img: NDArray) -> vtk.vtkDataArray:
    """
    Transforms a numpy array to a vtkDataArray.
    """
    vtk_array = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    return vtk_array

def build_transform_matrix(image: sitk.Image) -> NDArray:
    """
    Builds a 4x4 transformation matrix from a sitk image.
    """
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix


def export_sitk_to_vtk(
    sitk_img: sitk.Image,
    spacing: tuple[float, float, float] | None = None
) -> tuple[vtk.vtkImageData, vtk.vtkMatrix4x4]:
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
    """
    if not spacing:
        spacing = sitk_img.GetSpacing()
    img = sitk.GetArrayFromImage(sitk_img).transpose(2,1,0)
    vtk_array = export_numpy_to_vtk(img)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(sitk_img.GetSize())
    image_data.GetPointData().SetScalars(vtk_array)
    image_data.SetOrigin([0.,0.,0.])
    image_data.SetSpacing(spacing)
    matrix = build_transform_matrix(sitk_img)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_data)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    image_data = reslice.GetOutput()
    #imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return image_data, vtkmatrix


def numpy_to_points(vertices: NDArray, faces: NDArray, mesh_id: int = 0) -> vtk.vtkPolyData:
    """
    Converts a numpy array to a vtkPolyData.
    """
    poly = vtk.vtkPolyData()
    vtk_vertices = vtk.vtkPoints()
    vtk_vertices.SetData(numpy_to_vtk(vertices))

    vtk_faces = vtk.vtkCellArray()
    for i in range(faces.shape[0]):
        face = vtk.vtkTriangle()
        face.GetPointIds().SetId(0, faces[i][0])
        face.GetPointIds().SetId(1, faces[i][1])
        face.GetPointIds().SetId(2, faces[i][2])
        vtk_faces.InsertNextCell(face)

    poly.SetPolys(vtk_faces)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_vertices)
    poly.SetPolys(vtk_faces)

    arr = np.ones(poly.GetNumberOfPoints())*mesh_id
    arr_vtk = numpy_to_vtk(arr)
    arr_vtk.SetName('RegionId')
    poly.GetPointData().AddArray(arr_vtk)

    return poly


def convert_poly_data_to_image_data(poly: vtk.vtkPolyData, ref_im: vtk.vtkImageData) -> vtk.vtkImageData:
    """
    Convert the vtk polydata to imagedata
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """

    ref_im.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    return output


def center_crop(array: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
    """Perform center crop on a 3D array."""
    # Get current dimensions
    d, h, w = array.shape

    # Calculate start indices for the crop
    d_start = max(0, d // 2 - crop_size[0] // 2)
    h_start = max(0, h // 2 - crop_size[1] // 2)
    w_start = max(0, w // 2 - crop_size[2] // 2)

    # Perform the crop
    d_end = min(d_start + crop_size[0], d)
    h_end = min(h_start + crop_size[1], h)
    w_end = min(w_start + crop_size[2], w)

    cropped = array[d_start:d_end, h_start:h_end, w_start:w_end]

    # If the cropped array is smaller than the target size, pad it
    if cropped.shape != crop_size:
        result = np.zeros(crop_size, dtype=array.dtype)
        d_pad, h_pad, w_pad = cropped.shape

        # Calculate padding offsets to center the content
        d_offset = (crop_size[0] - d_pad) // 2
        h_offset = (crop_size[1] - h_pad) // 2
        w_offset = (crop_size[2] - w_pad) // 2

        # Place the cropped array in the center of the result array
        result[d_offset:d_offset+d_pad,
                h_offset:h_offset+h_pad,
                w_offset:w_offset+w_pad] = cropped
        return result

    return cropped


def preprocess(
    image: sitk.Image,
    window: float,
    level: float,
    crop_size: tuple[int, int, int]
) -> torch.Tensor:
    image_np = sitk.GetArrayFromImage(image)
    image_np = center_crop(image_np, crop_size)
    image_torch = torch.from_numpy(image_np).float()[None,None,]
    image_torch = apply_window_level(image_torch, window, level)
    return image_torch


def ensure_direction(
    image: sitk.Image,
    target_direction: tuple = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
) -> sitk.Image:
    """
    Ensures image data is oriented consistently with direction (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0).
    This standardizes the orientation so that the machine learning model receives consistent inputs.

    Args:
        image: SimpleITK image
        target_direction: The direction to ensure the image has

    Returns:
        SimpleITK image with normalized direction
    """
    # The target direction we want all images to match
    current_direction = image.GetDirection()

    if np.abs(np.array(current_direction)).sum() != 3:
        raise ValueError(f"Only axial images are supported, got {current_direction}")

    # If the image already has the target direction, return it unchanged
    if current_direction == target_direction:
        return image

    # Create a copy of the image to avoid modifying the original
    image_copy = sitk.Image(image)

    # Convert to numpy array for easier manipulation
    array = sitk.GetArrayFromImage(image_copy)

    # Check and flip each axis as needed
    # Direction matrix is 9 elements: [d11, d12, d13, d21, d22, d23, d31, d32, d33]
    # We care about the diagonal elements which tell us the orientation of each axis

    # X-axis: Check if d11 (element 0) differs from target
    if current_direction[0] != target_direction[0]:
        # Flip the X axis (3rd dimension in numpy array due to sitk/numpy dimension ordering)
        array = np.flip(array, axis=2)

    # Y-axis: Check if d22 (element 4) differs from target
    if current_direction[4] != target_direction[4]:
        # Flip the Y axis (2nd dimension in numpy array)
        array = np.flip(array, axis=1)

    # Z-axis: Check if d33 (element 8) differs from target
    if current_direction[8] != target_direction[8]:
        # Flip the Z axis (1st dimension in numpy array)
        array = np.flip(array, axis=0)

    # Create a new SimpleITK image from the modified array
    result = sitk.GetImageFromArray(array)

    # Copy metadata from original image
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(image.GetSpacing())
    # Set the direction to the target direction
    result.SetDirection(target_direction)

    return result


def restore_original_direction(
    image: sitk.Image,
    original_direction: tuple
) -> sitk.Image:
    """
    Restores an image processed with ensure_direction back to its original direction.

    Args:
        image: SimpleITK image to restore
        original_direction: The original direction of the input image

    Returns:
        SimpleITK image with original direction orientation
    """
    # The standard direction used during processing
    current_direction = image.GetDirection()

    # If directions are the same, no restoration needed
    if original_direction == current_direction:
        return image

    # Convert to numpy array for flipping operations
    array = sitk.GetArrayFromImage(image)

    # Reverse the flips that were applied in ensure_direction
    # X-axis: Check if d11 (element 0) differs from target
    if original_direction[0] != current_direction[0]:
        # Flip the X axis (3rd dimension in numpy array due to sitk/numpy dimension ordering)
        array = np.flip(array, axis=2)

    # Y-axis: Check if d22 (element 4) differs from target
    if original_direction[4] != current_direction[4]:
        # Flip the Y axis (2nd dimension in numpy array)
        array = np.flip(array, axis=1)

    # Z-axis: Check if d33 (element 8) differs from target
    if original_direction[8] != current_direction[8]:
        # Flip the Z axis (1st dimension in numpy array)
        array = np.flip(array, axis=0)

    # Create a new SimpleITK image from the modified array
    result = sitk.GetImageFromArray(array)

    # Copy metadata from original image
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(image.GetSpacing())
    # Set the direction to the original direction
    result.SetDirection(original_direction)

    return result


def restore_direction_mesh_seg(seg_array: np.ndarray, original_direction: tuple) -> np.ndarray:
    if original_direction[0] == -1.0:
        seg_array = np.flip(seg_array, axis=2)
    if original_direction[4] == -1.0:
        seg_array = np.flip(seg_array, axis=1)
    if original_direction[8] == -1.0:
        seg_array = np.flip(seg_array, axis=0)
    return seg_array


def main(
    implicitmeshnet_config: str,
    test_dir_in: str,
    test_dir_out: str,
    device: str = "cpu"
):
    with open(implicitmeshnet_config, "r", encoding="utf-8") as f:
        implicitmeshnet_config = json.load(f)
        cfg = ImplicitMeshNetTrainConfig(**implicitmeshnet_config)

    voxel2mesh_model = ShapeEstimationNet(
        unet_num_classes=cfg.unet_num_classes,
        gcn_out_channels=cfg.gcn_out_channels,
        gcn_amplification_factor=cfg.gcn_amplification_factor,
        unet_dropout_p=cfg.unet_dropout_p
    )

    snapshot = torch.load(cfg.snapshot_path, map_location="cpu", weights_only=False)
    voxel2mesh_model.load_state_dict(snapshot["voxel2mesh_state_dict"])
    voxel2mesh_model.to(device).eval()

    template_vertices, template_faces, template_edge_indices = get_mesh_template(
        mesh_template_radius=cfg.mesh_template_radius,
        mesh_template_subdivisions=cfg.mesh_template_subdivisions,
        device=device
    )

    os.makedirs(test_dir_out, exist_ok=True)
    files = os.listdir(test_dir_in)
    files = sorted(list(filter(lambda x: x.endswith(".nii.gz"), files)))
    for file in tqdm(files):
        # Read the original image
        input_path = os.path.join(test_dir_in, file)
        original_image = sitk.ReadImage(input_path)

        resampled_image = resample_image(original_image, cfg.target_spacing, sitk.sitkLinear)
        resampled_image_transposed = ensure_direction(resampled_image)

        # Proceed with normal preprocessing
        image = preprocess(resampled_image_transposed, cfg.window, cfg.level, cfg.crop_size)
        image = image.to(device)

        # get mesh from input image
        with torch.no_grad():
            meshes, segmentation = voxel2mesh_model(
                image,
                mesh_template=template_vertices,
                edge_index=template_edge_indices,
                get_segmentation=True
            )
            mesh = meshes[-1]  # [1, 2562, 3] with values in [-1, 1] corresponding to the cropped image
            segmentation = torch.argmax(segmentation, dim=1)[0].to(torch.int8).cpu().numpy()

            # Create a trimesh object with the transformed vertices
            output_mesh = trimesh.Trimesh(
                vertices=mesh[0].cpu().numpy(),
                faces=template_faces.cpu().numpy(),
                process=False
            )

            # fix vertices to match the original image direction
            direction_matrix_original = np.array(original_image.GetDirection()).reshape(3, 3)
            direction_matrix_resampled = np.array(resampled_image_transposed.GetDirection()).reshape(3, 3)
            direction_matrix = direction_matrix_original @ direction_matrix_resampled
            direction_matrix = np.rot90(np.rot90(direction_matrix))  # swap x and z
            output_mesh.vertices = (direction_matrix @ output_mesh.vertices.T).T

            # undo normalization from [-1, 1] to [0, 1]
            output_mesh.vertices = (output_mesh.vertices + 1) / 2

            # convert vertices from xyz to dhw
            output_mesh.vertices = output_mesh.vertices[:, ::-1]

            # convert from [0, 1] to voxel indices
            output_mesh.vertices = output_mesh.vertices * np.array([
                cfg.crop_size[0], cfg.crop_size[1], cfg.crop_size[2]
            ])

            # Step 2: Account for center cropping
            # Get resampled image size before cropping
            resampled_array_shape = np.array(sitk.GetArrayFromImage(resampled_image_transposed).shape)

            # Calculate crop offset
            d_start = max(0, resampled_array_shape[0] // 2 - cfg.crop_size[0] // 2)
            h_start = max(0, resampled_array_shape[1] // 2 - cfg.crop_size[1] // 2)
            w_start = max(0, resampled_array_shape[2] // 2 - cfg.crop_size[2] // 2)

            # Calculate actual cropped size (might be smaller than crop_size)
            d_pad = min(cfg.crop_size[0], resampled_array_shape[0] - d_start)
            h_pad = min(cfg.crop_size[1], resampled_array_shape[1] - h_start)
            w_pad = min(cfg.crop_size[2], resampled_array_shape[2] - w_start)

            # Calculate padding offsets
            d_offset = (cfg.crop_size[0] - d_pad) // 2
            h_offset = (cfg.crop_size[1] - h_pad) // 2
            w_offset = (cfg.crop_size[2] - w_pad) // 2

            # First subtract the padding offset (since vertices are in the padded space)
            output_mesh.vertices[:, 0] -= w_offset  # Subtract width padding
            output_mesh.vertices[:, 1] -= h_offset  # Subtract height padding
            output_mesh.vertices[:, 2] -= d_offset  # Subtract depth padding

            # Then add the crop offset (to place in resampled image space)
            output_mesh.vertices[:, 0] += w_start  # Add width offset
            output_mesh.vertices[:, 1] += h_start  # Add height offset
            output_mesh.vertices[:, 2] += d_start  # Add depth offset

            for i in range(output_mesh.vertices.shape[0]):
                output_mesh.vertices[i] = resampled_image.TransformContinuousIndexToPhysicalPoint(
                    output_mesh.vertices[i]
                )

            # Calculate actual cropped size (might be smaller than crop_size)
            d_pad = min(cfg.crop_size[0], resampled_array_shape[0] - d_start)
            h_pad = min(cfg.crop_size[1], resampled_array_shape[1] - h_start)
            w_pad = min(cfg.crop_size[2], resampled_array_shape[2] - w_start)

            # Calculate padding offsets for the segmentation
            d_offset = (cfg.crop_size[0] - d_pad) // 2
            h_offset = (cfg.crop_size[1] - h_pad) // 2
            w_offset = (cfg.crop_size[2] - w_pad) // 2

            segmentation_np = np.zeros(resampled_array_shape, dtype=np.ubyte)
            segmentation_np[d_start:d_start+d_pad,
                          h_start:h_start+h_pad,
                          w_start:w_start+w_pad] = segmentation[d_offset:d_offset+d_pad,
                                                                h_offset:h_offset+h_pad,
                                                                w_offset:w_offset+w_pad]

            # Save the mesh
            mesh_output_path = os.path.join(test_dir_out, file.replace("_image.nii.gz", "_mesh.ply"))
            output_mesh.export(mesh_output_path)
            segmentation_itk = sitk.GetImageFromArray(segmentation_np)
            segmentation_itk.CopyInformation(resampled_image_transposed)
            segmentation_itk = restore_original_direction(segmentation_itk, original_image.GetDirection())

            # save segmentation
            sitk.WriteImage(
                segmentation_itk,
                os.path.join(test_dir_out, file.replace("_image.nii.gz", "_unet_segmentation.nii.gz"))
            )

            poly = numpy_to_points(output_mesh.vertices, output_mesh.faces, mesh_id=1)
            img_vtk, _ = export_sitk_to_vtk(original_image)
            img_poly = convert_poly_data_to_image_data(poly, img_vtk)
            seg_array = convert_vtk_image_to_numpy(img_poly)

            seg_itk = sitk.GetImageFromArray(
                restore_direction_mesh_seg(seg_array.transpose(2, 1, 0), original_image.GetDirection())
            )
            seg_itk = sitk.Cast(seg_itk, sitk.sitkUInt8)
            seg_itk.CopyInformation(original_image)
            seg_itk = restore_original_direction(seg_itk, original_image.GetDirection())
            sitk.WriteImage(seg_itk, os.path.join(test_dir_out, file.replace("_image.nii.gz", "_mesh_labelmap.nii.gz")))


if __name__ == "__main__":
    Fire(main)
