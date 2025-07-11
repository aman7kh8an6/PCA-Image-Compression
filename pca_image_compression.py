"""
Created on Thus Jul 10
@author: Aman Khan
PCA Image Compression using orthogonal Iteration
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple


def load_image(filepath: str) -> np.ndarray:
    """
    Loads an image, converts it to grayscale, and returns it as a NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        The loaded image as a NumPy array.
    """
    try:
        img = Image.open(filepath).convert('L')  # 'L' mode is for grayscale
        return np.array(img, dtype=float)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at: {filepath}")


def image_to_blocks(image: np.ndarray, block_size: Tuple[int, int]) -> np.ndarray:
    """
    Divides an image into non-overlapping blocks.

    Parameters
    ----------
    image : np.ndarray
        The input image with shape (height, width).
    block_size : tuple[int, int]
        The (height, width) of the blocks.

    Returns
    -------
    np.ndarray
        A 3D array of shape (num_blocks, block_height, block_width).
    """
    img_height, img_width = image.shape
    block_height, block_width = block_size

    # Calculate how many blocks fit in each dimension
    num_blocks_h = img_height // block_height
    num_blocks_w = img_width // block_width

    # Crop the image to be divisible by the block size
    image_cropped = image[:num_blocks_h *
                          block_height, :num_blocks_w * block_width]

    # Reshape the image into blocks
    # Resulting shape is (num_blocks_h, num_blocks_w, block_height, block_width)
    blocks = image_cropped.reshape(
      num_blocks_h, block_height, num_blocks_w, block_width).transpose(0, 2, 1, 3)

    # Reshape into a 3D array of blocks
    # Resulting shape is (num_blocks, block_height, block_width)
    return blocks.reshape(-1, block_height, block_width)


def blocks_to_image(blocks: np.ndarray, image_shape: Tuple[int, int],
                    block_size: Tuple[int, int]) -> np.ndarray:
    """
    Reassembles an image from its blocks.

    Parameters
    ----------
    blocks : np.ndarray
        The array of image blocks with shape (num_blocks, block_height, block_width).
    image_shape : tuple[int, int]
        The shape (height, width) of the original (cropped) image.
    block_size : tuple[int, int]
        The (height, width) of the blocks.

    Returns
    -------
    np.ndarray
        The reconstructed image with shape image_shape.
    """
    num_blocks_h = image_shape[0] // block_size[0]
    num_blocks_w = image_shape[1] // block_size[1]

    # Reshape blocks back to the 4D structure
    blocks_4d = blocks.reshape(
      num_blocks_h, num_blocks_w, block_size[0], block_size[1])

    # Transpose and reshape to form the final image
    return blocks_4d.transpose(0, 2, 1, 3).reshape(image_shape)


# --------------------------------------------------------------------------
# CORE PCA IMPLEMENTATION (TODO: YOUR CODE HERE)
# --------------------------------------------------------------------------
def orthogonal_iteration(A: np.ndarray, p: int, maxIter: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Orthogonal Iteration to find the p dominant eigenvectors of a matrix A.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix.
    p : int
        The number of dominant eigenvectors to find.
    maxIter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Tolerance for convergence checking (for eigenvalues), by default 1e-12

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing: - An orthonormal matrix whose columns are the approximate eigenvectors.
                            - The upper triangular matrix from the last QR decomposition.
    """
    n = A.shape[0]
    # Step 1: Start with a random orthonormal matrix Q0 (n x p)
    Q, _ = np.linalg.qr(np.random.randn(n, p))
    
    for _ in range(maxIter):
        Z = A @ Q
        Q_new, _ = np.linalg.qr(Z)
        T = Q_new.T @ A @ Q_new

        # Convergence check: A Q ≈ Q T
        residual = np.linalg.norm(A @ Q_new - Q_new @ T, ord='fro')
        if residual < tol:
            return Q_new, T

        Q = Q_new

    return (Q, T)


def calculate_pca(data: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the principal components of the given data.

    Parameters
    ----------
    data : np.ndarray
        The input data matrix of shape (num_samples, num_features).
        For our image blocks, this will be (num_blocks, 64).
    p : int
        Number of principal components to compute.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
            - eigenvalues: 1D array of dominant eigenvalues.
            - eigenvectors: 2D array where each column is an eigenvector,
                         corresponding to the dominant eigenvalues.
            - mean_vector: 1D array representing the mean of the data.
    """
    m,n_features = data.shape
    
    if p > n_features:
        raise ValueError(f"p = {p} cannot be greater than number of features = {n_features}")
    if m < 2:
        raise ZeroDivisionError("Need at least two data points to compute covariance. ")

    #Centre Data
    mean_vector = np.mean(data, axis = 0)
    centred_data = data - mean_vector

    #Calculating Covariance Matrix
    S = (centred_data.T @ centred_data)/(m-1)
    eigenvectors, T = orthogonal_iteration(S, p)
    eigenvalues = np.diagonal(T)
    
    #Sort by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return (eigenvalues, eigenvectors,mean_vector)


def compress(data: np.ndarray, eigenvectors: np.ndarray,
             mean_vector: np.ndarray) -> np.ndarray:
    """
    Compresses the data by projecting it onto the top p principal components.

    Parameters
    ----------
    data : np.ndarray
        The original data with shape (num_samples, num_features).
    eigenvectors : np.ndarray
        The (reduced) set of sorted eigenvectors with shape (num_features, p).
    mean_vector : np.ndarray
        The mean vector of the original data with shape (num_features,).

    Returns
    -------
    np.ndarray
        The compressed data with shape (num_samples, p).
    """
    centered_data = data - mean_vector
    compressed_data = centered_data @ eigenvectors
    return compressed_data


def decompress(data: np.ndarray, eigenvectors: np.ndarray,
               mean_vector: np.ndarray) -> np.ndarray:
    """
    Decompresses the data by back-projecting it from the reduced component space.

    Parameters
    ----------
    data : np.ndarray
        The compressed data with shape (num_samples, p).
    eigenvectors : np.ndarray
        The (reduced) set of sorted eigenvectors with shape (num_features, p).
    mean_vector : np.ndarray
        The mean vector of the original data with shape (num_features,).

    Returns
    -------
    np.ndarray
        The reconstructed data with shape (num_samples, num_features).
    """
    reconstructed_data = data @ eigenvectors.T
    reconstructed_data += mean_vector
    return reconstructed_data


if __name__ == "__main__":
    IMAGE_FILEPATH = '/Users/amankhan/Downloads/ChatGPTImage.png'
    original_image = load_image(IMAGE_FILEPATH)

    BLOCK_SIZE = (8, 8)
    # P_COMPONENTS is the number of principal components to keep. Max is 8*8=64.
    # Once the workflow is established experiment with different values (e.g., 1, 4, 8, 16, 32, 64).
    P_COMPONENTS = 8
    if P_COMPONENTS > np.prod(BLOCK_SIZE):
        raise ValueError(
          "P_COMPONENTS cannot be larger than the number of elements in a block.")

    # --- 1. Load and Prepare Data ---
    image_blocks = image_to_blocks(original_image, BLOCK_SIZE)

    # get Number of blocks or number of samples, height, width
    num_blocks, block_h, block_w = image_blocks.shape
    num_features = block_h * block_w  # Number of features per block, here 8*8=64
    # Reshape to (num_samples, num_features)
    data_matrix = image_blocks.reshape(num_blocks, num_features)

    print(
      f"Image loaded and divided into {num_blocks} blocks of size {BLOCK_SIZE}.")
    print(f"Data matrix shape: {data_matrix.shape}")

    # --- 2. Perform PCA ---
    print("\nCalculating PCA...")
    eigenvalues, eigenvectors, mean_vector = calculate_pca(
      data_matrix, P_COMPONENTS)
    print("PCA calculation complete.")

    # --- 3. Compress and Decompress ---
    print(f"\nCompressing data using top {P_COMPONENTS} components...")
    compressed_data = compress(
      data_matrix, eigenvectors, mean_vector)
    print(f"Compressed data shape: {compressed_data.shape}")

    print("\nDecompressing data...")
    reconstructed_data = decompress(
      compressed_data, eigenvectors, mean_vector)
    print(f"Reconstructed data shape: {reconstructed_data.shape}")

    # --- 4. Reconstruct and Display Image ---
    # Reshape back to (num_samples, block_h, block_w)
    reconstructed_blocks = reconstructed_data.reshape(
      num_blocks, block_h, block_w)

    # During image_to_block we cropped the image to be divisible by the block size
    # So we can only get the cropped image back
    cropped_shape = (original_image.shape[0] - original_image.shape[0] %
                     block_h, original_image.shape[1] - original_image.shape[1] % block_w)

    reconstructed_image = blocks_to_image(
      reconstructed_blocks, cropped_shape, BLOCK_SIZE)  # Reconstruct image

    # --- 5. Visualization ---
    original_size = data_matrix.nbytes  # Original size in bytes
    # Compressed size includes coefficients, the eigenvectors, and the mean vector
    # Get Compressed size in bytes
    compressed_size = compressed_data.nbytes + \
        eigenvectors.nbytes + mean_vector.nbytes
    compression_ratio = original_size / compressed_size  # Compression ratio

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
      f'Reconstructed (p={P_COMPONENTS})\nCompression Ratio: {compression_ratio:.1f}x')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()