import os
import random
from PIL import Image
import collections

import submodule_utils.image.preprocess as preprocess

class SlideCoordsExtractor(collections.abc.Sequence):
    """Iterable that tiles the OpenSlide slide image with adjacent non-overlapping patch tiles of size patch_size. It does not extract tile to a PIL image. It only returns the tile coordinates of the patches.
    """
    def __init__(self, os_slide, patch_size, shuffle=False, seed=1):
        """
        Parameters
        ----------
        os_slide : OpenSlide
            OpenSlide slide to get patch coordinates from.
        
        patch_size : int
            The size of the patch to extract.

        shuffle : bool
            Whether to shuffle coordinates to extract. 
        
        seed : int
            Seed for shuffle.
        """
        self.os_slide = os_slide
        self.patch_size = patch_size
        self.width, self.height = self.os_slide.dimensions
        self.tile_width = int(self.width / self.patch_size)
        self.tile_height = int(self.height / self.patch_size)
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.indices = list(range(self.tile_width * self.tile_height))
            random.seed(seed)
            random.shuffle(self.indices)

    def __len__(self):
        if self.shuffle:
            return len(self.indices)
        else:
            return self.tile_width * self.tile_height
    
    def __getitem__(self, idx):
        """Get tile coordinate from index.

        Parameters
        ----------
        idx : int
            Index to get tile coordinate.

        Returns
        -------
        tuple
            A tuple of
            - tile_x (int)
            - tile_y (int) where tile_x and tile_y are the row and columns respectively of the patch_size by patch_size grid on the slide image where the image is extracted
            - x (int)
            - y (int) where x and y are pixel coordinates of slide image and (x,y) = (0,0) is the top left corner of the image.
        """
        if idx >= len(self):
            raise IndexError
        if self.shuffle:
            idx = self.indices[idx]
        tile_x = idx % self.tile_width
        tile_y = int(idx / self.tile_width)
        x = tile_x * self.patch_size
        y = tile_y * self.patch_size
        return (tile_x, tile_y, x, y,)


class SlidePatchExtractor(SlideCoordsExtractor):
    def __init__(self, os_slide, patch_size, resize_sizes=None, shuffle=False, seed=1):
        """Iterable that tiles the OpenSlide slide image with adjacent non-overlapping patch tiles of size patch_size, extracts each tile to a PIL image, and then resizes that tile by each resize size in resize_sizes. The patch image, the tile coordinate, and the patch's resized images are returned.

        Parameters
        ----------
        os_slide : OpenSlide
            OpenSlide slide to extract patches from

        patch_size : int
            The size of the patch to extract
        
        resize_sizes : (list of int) or None
            A list of multiple sizes to resize. Each size must be at most patch_size.

        Returns
        -------
        tuple
            A tuple containing
             - patch (Pillow.Image) patch extracted using coordinates from SlideCoordsExtractor.
             - tile_x, tile_y, x, y (tuple of int) The coordinates returned from SlideCoordsExtractor.
             - resized_patches (dict of int: Image) A dictionary where key is one of patch_size and resize_sizes, and value is patch downsampled to size specified in key.
        """
        super().__init__(os_slide, patch_size, shuffle=shuffle, seed=seed)
        try:
            self.resize_sizes = resize_sizes.copy()
            if patch_size not in self.resize_sizes:
                self.resize_sizes.insert(0, patch_size)
        except:
            self.resize_sizes = None

    def __getitem__(self, idx):
        tile_x, tile_y, x, y = super().__getitem__(idx)
        patch = preprocess.extract(self.os_slide, x, y, self.patch_size)
        if self.resize_sizes:
            resized_patches = { }
            for resize_size in self.resize_sizes:
                if resize_size == self.patch_size:
                    resized_patches[resize_size] = patch
                else:
                    resized_patches[resize_size] = preprocess.resize(patch, resize_size)
            return patch, (tile_x, tile_y, x, y,), resized_patches
        else:
            return patch, (tile_x, tile_y, x, y,), { self.patch_size: patch }
