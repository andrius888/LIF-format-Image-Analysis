from readlif.reader import LifFile
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation
import scipy.ndimage as ndi
import skimage.measure as skm



def import_lif_file(lif_file_path=None):
    """
    Imports a LIF file and returns a list of images contained within the file.

    Parameters:
    lif_file_path (str): Path to the LIF file. If None, prompts for input.

    Returns:
    list: A list of dictionaries, each containing data and metadata for one image in the LIF file.
    """
    if lif_file_path is None:
        print("Input the LIF file path:")
        lif_file_path = input().strip('"')
    lif_file = LifFile(lif_file_path)

    img_list = [i for i in lif_file.get_iter_image()]
    all_images = []
    num_channels = None
    num_z_stacks = None

    for image_index, current_image in enumerate(img_list):
        channel_images = []
        variable_names = []

        if image_index == 0:
            channel_index = 0
            try:
                while True:
                    try:
                        current_frame = np.array(current_image.get_frame(z=0, t=0, c=channel_index))
                        channel_index += 1
                    except ValueError as e:
                        if "Requested channel doesn't exist" in str(e):
                            break
            except IndexError:
                pass
            num_channels = channel_index

            z_index = 0
            try:
                while True:
                    try:
                        current_frame = np.array(current_image.get_frame(z=z_index, t=0, c=0))
                        z_index += 1
                    except ValueError as e:
                        if "Requested Z frame doesn't exist" in str(e):
                            break
            except IndexError:
                pass
            num_z_stacks = z_index

        image_dimension = "2D" if num_z_stacks == 1 else "3D"

        for channel_index in range(num_channels):
            channel_frames = []
            for z_index in range(num_z_stacks):
                try:
                    current_frame = np.array(current_image.get_frame(z=z_index, t=0, c=channel_index))
                    channel_frames.append(current_frame)
                except ValueError as e:
                    if "Requested Z frame doesn't exist" in str(e):
                        break
            variable_name = f"{current_image.name}_channel_{channel_index + 1}"
            globals()[variable_name] = channel_frames
            channel_images.append(channel_frames)
            variable_names.append(variable_name)

        combined_image_object = {
            "image_name": current_image.name.replace(" ", "_"),
            "channel_images": channel_images,
            "variable_names": variable_names,
            "image_dimension": image_dimension
        }

        all_images.append(combined_image_object)

    list_name = f"all_images_{len(all_images)}_images"
    print(f"List named {list_name} was created for {image_dimension} images")

    return all_images




def view_2d_images(image_list):
    """
    Displays 2D images from a list of image objects, each containing multiple channels.

    Parameters:
    image_list (list): List of image objects to be displayed.
    """
    for image_obj in image_list:
        combined_images = [np.dstack(channel_frames) for channel_frames in image_obj["channel_images"]]

        num_channels = len(image_obj["channel_images"])
        fig, axs = plt.subplots(1, num_channels, figsize=(8 * num_channels, 6))

        for channel_index in range(num_channels):
            axs[channel_index].imshow(combined_images[channel_index][:, :, 0])
            axs[channel_index].set_title(f'Channel {channel_index + 1}')

        fig.suptitle(f"{image_obj['image_name']} ({image_obj['image_dimension']})")
        plt.tight_layout()
        plt.show()
        
        
        
        

def quick_view_3d_images(image_list):
    """
    Provides a quick view of 3D images from a list of image objects, displaying the middle stack.

    Parameters:
    image_list (list): List of 3D image objects to be displayed.
    """
    for image_obj in image_list:
        combined_images = [np.dstack(channel_frames) for channel_frames in image_obj["channel_images"]]

        middle_stack_index = len(image_obj["channel_images"][0]) // 2 if len(image_obj["channel_images"][0]) > 1 else 0

        num_channels = len(image_obj["channel_images"])
        fig, axs = plt.subplots(1, num_channels, figsize=(8 * num_channels, 6))

        for channel_index in range(num_channels):
            axs[channel_index].imshow(combined_images[channel_index][:, :, middle_stack_index])
            axs[channel_index].set_title(f'Channel {channel_index + 1}')

        fig.suptitle(f"{image_obj['image_name']} ({image_obj['image_dimension']}) - Middle Stack")
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
        
def display_all_3Dlayers(image_array):
    """
    Displays all layers of a 3D image array in a grid of subplots.

    Parameters:
    image_array (numpy.ndarray): A 3D numpy array representing an image.
    """

    num_layers = len(image_array)
    num_columns = int(np.ceil(np.sqrt(num_layers)))
    num_rows = int(np.ceil(num_layers / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_layers):
        ax = axes[i]
        ax.imshow(image_array[i])
        ax.axis('off')
        ax.set_title(f'Layer {i+1}')

    for i in range(num_layers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
def display_all_3Dlayers_from_list(all_images):
    """
    Displays all the layers of all the channels of every image in all_images list.
    
    Parameters:
    all_images (list): List containing image data and metadata.
    """
    for image in all_images:
        image_name = image['image_name']
        channel_images = image['channel_images']
        variable_names = image['variable_names']

        for channel_index, channel in enumerate(channel_images):
            print(f"Displaying {variable_names[channel_index]} of {image_name}")
            display_all_3Dlayers(channel)
    
    
    
def count_dapi(image_name, dapi_img):
    """
    Counts DAPI-stained cells in an image and displays the image with the count.

    Parameters:
    image_name (str): Name of the image.
    dapi_image (numpy.ndarray): Numpy array representation of the image.

    Returns:
    dict: A dictionary with the image name as key and cell count as value.
    """
    elevation_map = filters.sobel(dapi_img)
    markers = np.zeros_like(dapi_img)
    markers[dapi_img < 3] = 1
    markers[dapi_img > 18] = 2
    segmentation_img = segmentation.watershed(elevation_map, mask=markers)
    
    fill = ndi.binary_fill_holes(segmentation_img)
    label_cleaned = morphology.remove_small_objects(fill, 50)
    label_cleaned = ndi.binary_fill_holes(label_cleaned)
    labeled_dapi, _ = ndi.label(label_cleaned)
    cell_label = skm.regionprops(labeled_dapi)
    cell_count = len(cell_label)
    
    plt.imshow(dapi_img)
    plt.title(f"{image_name} - Cell Count: {cell_count}")
    plt.show()
    
    plt.imshow(label_cleaned)
    plt.show()

    return {image_name: cell_count}





def pixel_intensity_calc(image_name, np_image):
    """
    Calculates the mean pixel intensity of an image and displays the image with this value.

    Parameters:
    image_name (str): Name of the image.
    np_image (numpy.ndarray): Numpy array representation of the image.

    Returns:
    dict: A dictionary with the image name as key and mean intensity as value.
    """
    mean_intensity = np.mean(np_image)
    plt.imshow(np_image)
    plt.title(f"{image_name} - Mean Intensity: {mean_intensity:.2f}")
    plt.show()
    return {image_name: mean_intensity}





def cell_counter_3D(image_name, np_image, min_cell_size=1000, max_cell_size=6000):
    """
    Count cells in a 3D z-stack image using advanced image processing techniques.
    
    Parameters:
    image_name (str): Name of the image for display purposes.
    np_image (numpy.ndarray): A 3D numpy array representing z-stack image.
    min_cell_size (int): Minimum size of the cell to be considered valid.
    max_cell_size (int): Maximum size of the cell to be considered valid.
    
    Returns:
    int: The number of cells counted in the 3D image.
    """
    filtered_image = ndi.median_filter(np_image, size=3)
    blurred_image = ndi.gaussian_filter(filtered_image, sigma=2)
    threshold_value = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > threshold_value
    filled_image = ndi.binary_fill_holes(binary_image)
    distance_map = ndi.distance_transform_edt(filled_image)
    local_maxi = morphology.local_maxima(distance_map)
    markers = ndi.label(local_maxi)[0]
    labels_ws = segmentation.watershed(-distance_map, markers, mask=filled_image)
    labeled_cells = measure.label(labels_ws)
    cell_sizes = ndi.sum(filled_image, labeled_cells, range(np.max(labeled_cells) + 1))
    mask = (cell_sizes > min_cell_size) & (cell_sizes < max_cell_size)
    cleaned_cells = mask[labeled_cells]
    cell_count = np.sum(mask) - 1

    max_proj_original = np.max(np_image, axis=0)
    labeled_cells_masked = labeled_cells * mask[labeled_cells]
    max_proj_labeled = np.max(labeled_cells_masked, axis=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(max_proj_original, cmap='gray')
    plt.title(f"Original Image: {image_name}\nCell Count: {cell_count}")

    plt.subplot(1, 2, 2)
    plt.imshow(max_proj_labeled, cmap='nipy_spectral')
    plt.title("Segmented Cells (3D)")
    
    plt.show()

    return cell_count