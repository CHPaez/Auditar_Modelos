import numpy as np
from PIL import Image

from robustness_test import perturb


def test_perturb_preserves_image_size():
    image = Image.new("RGB", (64, 48), color=(120, 200, 80))

    result = perturb(image)

    assert isinstance(result, Image.Image)
    assert result.size == image.size


def test_perturb_changes_pixel_values():
    image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))

    result = perturb(image)

    assert np.array(result).shape == (64, 64, 3)
    # blur + rotation + brightness/contrast noise should change at least some pixels
    assert not np.array_equal(np.array(result), np.array(image))
