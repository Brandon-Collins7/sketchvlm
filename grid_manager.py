"""
Grid Manager - Handles dynamic grid sizing and image placement for sketch annotations.
"""

from typing import Tuple, Dict, Optional
from PIL import Image, ImageOps
import utils
import math

class GridManager:
    """Manages dynamic grid creation and image placement for sketch annotations."""
    
    def __init__(self, cell_size: int = 15, min_grid: int = 10, max_grid: int = 100):
        """
        Initialize the GridManager.
        
        Args:
            cell_size: Size of each grid cell in pixels
            min_grid: Minimum grid size
            max_grid: Maximum grid size
        """
        self.cell_size = cell_size
        self.min_grid = min_grid
        self.max_grid = max_grid
        
        # Current grid state
        self.res_x = 50  # Default grid columns
        self.res_y = 50  # Default grid rows
        self.grid_size = (765, 765)  # Default canvas size
        self.grid_image = None
        self.positions = {}
        
    def calculate_grid_size_for_image(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate optimal grid size to fit around an image at its natural size.
        Image will be positioned at bottom-left corner of the grid.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            tuple: (grid_res_x, grid_res_y, grid_width, grid_height)
        """
        # Calculate exact cells needed to fit the image (round up fractional cells)
        min_cells_x = (image_width + self.cell_size - 1) // self.cell_size  # Ceiling division
        min_cells_y = (image_height + self.cell_size - 1) // self.cell_size  # Ceiling division
        
        # Use exact cells needed, no extra padding
        grid_res_x = max(self.min_grid, min(self.max_grid, min_cells_x))
        grid_res_y = max(self.min_grid, min(self.max_grid, min_cells_y))
        
        # utils.py adds +1 for header automatically, so final grid dimensions are:
        grid_width = (grid_res_x + 1) * self.cell_size  # utils.py calculation
        grid_height = (grid_res_y + 1) * self.cell_size  # utils.py calculation
        
        return grid_res_x, grid_res_y, grid_width, grid_height
    
    def update_grid_for_image(self, img, show_full_grid=False):
        W, H = img.size
        old = (self.res_x, self.res_y, self.grid_size)
        self._recompute_grid(W, H, show_full_grid)
        return (self.res_x, self.res_y, self.grid_size) != old

    
    def place_image_on_canvas(self, img: Image.Image, bgcolor=(255, 255, 255)) -> Image.Image:
        """
        Place image at its natural size at the bottom-left corner of the grid canvas.
        No scaling or stretching - just positioning the original image.
        
        Args:
            img: PIL Image to place
            bgcolor: Background color for the canvas
            
        Returns:
            Image with the input image positioned on the grid canvas
        """
        img = ImageOps.exif_transpose(img)  # respect EXIF orientation
        W, H = self.grid_size
        
        # Create canvas
        canvas = Image.new("RGB", (W, H), bgcolor)
        
        # Position image at bottom-left corner (aligned with grid)
        x = self.cell_size  # Start right after the header column
        y = H - self.cell_size - img.height  # Bottom aligned (above bottom header row)
        
        # Paste the image at its natural size
        canvas.paste(img.convert("RGB"), (x, y))
        return canvas
    
    def overlay_grid(self, img_rgb: Image.Image) -> Image.Image:
        """
        Overlay the current grid on top of an image.
        
        Args:
            img_rgb: RGB image to overlay grid on
            
        Returns:
            Image with grid overlaid
        """
        if self.grid_image is None:
            return img_rgb
            
        grid = self.grid_image.convert("RGBA")
        base = img_rgb.convert("RGBA")
        # Keep only the grid lines (white/transparent background)
        mask = grid.convert("L").point(lambda p: 255 if p < 200 else 0)
        base.paste(grid, (0, 0), mask)
        return base.convert("RGB")
    
    def create_annotated_image(self, img, show_full_grid=False, bgcolor=(255,255,255)):
        self.update_grid_for_image(img, show_full_grid)   # ← this is the “recompute”
        canvas = self.place_image_at_bottom_left(img, bgcolor)  # or whatever you named it
        return self.overlay_grid(canvas)

    def _recompute_grid(self, W: int, H: int, show_full_grid: bool):
        # derive grid cell counts from raw image size
        self.res_x = math.ceil(W / self.cell_size)
        self.res_y = math.ceil(H / self.cell_size)

        # +1 cell for the left label column, +1 cell for the bottom label row
        self.grid_size = (
            self.res_x * self.cell_size + self.cell_size,
            self.res_y * self.cell_size + self.cell_size,
        )

        # (re)build the labeled grid + positions
        self.grid_image, self.positions = utils.create_grid_image(
            res_x=self.res_x,
            res_y=self.res_y,
            cell_size=self.cell_size,
            header_size=self.cell_size,
            full=show_full_grid,
        )

    def place_image_at_bottom_left(self, img, bgcolor=(255,255,255)):
        canvas = Image.new("RGB", self.grid_size, bgcolor)
        # top-aligned, left-shifted by one cell to leave the left label column,
        # and leaves the bottom header row visible
        canvas.paste(img.convert("RGB"), (self.cell_size, 0))
        return canvas

    
    def get_grid_info(self) -> Dict:
        """
        Get current grid information.
        
        Returns:
            Dictionary with grid dimensions and settings
        """
        return {
            "res_x": self.res_x,
            "res_y": self.res_y,
            "grid_size": self.grid_size,
            "cell_size": self.cell_size,
            "total_cells": self.res_x * self.res_y,
            "positions_count": len(self.positions)
        }